/*
 * Hybrid Parallel Document Similarity Search
 * ============================================
 * Parallelism model : MPI (inter-node) + OpenMP (intra-node)
 *   - MPI distributes document partitions across ranks
 *   - OpenMP parallelises the similarity computation within each rank
 *
 * Key algorithmic choices:
 *   - MPI_Bcast  : broadcast query vector and metadata
 *   - MPI_Scatterv: distribute variable-length document blocks
 *   - Min-heap   : O(n log k) local top-k (vs O(n log n) full sort)
 *   - MPI_Gather : collect per-rank top-k to master
 *   - Merge sort : O(P·k log P·k) final global merge on master
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>
#include "utils.h"

/* ─────────────────────────────────────────────────────────────────
 * kreduce  –  gather every rank's top-k to master and pick global top-k
 *
 * FIX over original: every rank always sends exactly `top_k` DocScores.
 * If a rank has fewer local documents we pad with sentinel entries
 * (score = -DBL_MAX, id = -1) so MPI_Gather can use a uniform count.
 * The master filters sentinels before the final sort.
 * ───────────────────────────────────────────────────────────────── */
static void kreduce(DocScore *local_topk, int actual_k, int top_k,
                    int world_size, int rank, DocScore *global_topk) {

    /* Build a padded buffer of exactly top_k entries */
    DocScore *padded = (DocScore *)malloc(top_k * sizeof(DocScore));
    if (!padded) { MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int i = 0; i < top_k; i++) {
        if (i < actual_k && local_topk) {
            padded[i] = local_topk[i];
        } else {
            padded[i].id    = -1;
            padded[i].score = -DBL_MAX;   /* sentinel – filtered on master */
        }
    }

    DocScore *all_topk = NULL;
    if (rank == 0) {
        all_topk = (DocScore *)malloc(world_size * top_k * sizeof(DocScore));
        if (!all_topk) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    /* All ranks send exactly top_k DocScores → uniform count → correct gather */
    MPI_Gather(padded,   top_k * sizeof(DocScore), MPI_BYTE,
               all_topk, top_k * sizeof(DocScore), MPI_BYTE,
               0, MPI_COMM_WORLD);

    free(padded);

    if (rank == 0) {
        /* Filter sentinels, sort descending, take first top_k */
        int total   = world_size * top_k;
        int valid   = 0;

        /* Compact in-place: move valid entries to front */
        for (int i = 0; i < total; i++) {
            if (all_topk[i].id != -1)
                all_topk[valid++] = all_topk[i];
        }

        mergeSortDocScores(all_topk, 0, valid - 1);

        int out_k = (top_k < valid) ? top_k : valid;
        memcpy(global_topk, all_topk, out_k * sizeof(DocScore));

        /* Zero-out any unfilled slots (edge case: fewer total docs than top_k) */
        for (int i = out_k; i < top_k; i++) {
            global_topk[i].id    = -1;
            global_topk[i].score = -DBL_MAX;
        }

        free(all_topk);
    }
}

/* ═══════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    /* ── MPI initialisation ── */
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t_total_start = MPI_Wtime();

    /* ── Argument parsing (rank 0 validates, aborts all on error) ── */
    if (argc < 5) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int   dict_size      = atoi(argv[1]);
    int   top_k          = atoi(argv[2]);
    char *documents_file = argv[3];
    char *query_file     = argv[4];

    SimilarityMetric metric = SIMILARITY_POWER;
    if (argc >= 6) {
        if      (strcmp(argv[5], "cosine")    == 0) metric = SIMILARITY_COSINE;
        else if (strcmp(argv[5], "euclidean") == 0) metric = SIMILARITY_EUCLIDEAN;
        else if (strcmp(argv[5], "manhattan") == 0) metric = SIMILARITY_MANHATTAN;
    }

    /* Optional: explicit OpenMP thread count per MPI rank */
    int num_threads = 1;
    if (argc >= 7) num_threads = atoi(argv[6]);
    omp_set_num_threads(num_threads);

    if (dict_size <= 0 || top_k <= 0) {
        if (rank == 0) fprintf(stderr, "Error: dict_size and top_k must be > 0\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ── Master: file I/O ── */
    double t_io_start = 0, t_io_end = 0;
    int    total_docs    = 0;
    int   *all_documents = NULL;
    int   *all_ids       = NULL;
    int   *query_arr     = (int *)malloc(dict_size * sizeof(int));

    if (rank == 0) {
        t_io_start = MPI_Wtime();

        const char *metric_name[] = {"Power-based","Cosine","Euclidean","Manhattan"};
        printf("=== Hybrid Parallel Document Similarity Search ===\n");
        printf("MPI ranks    : %d\n",  world_size);
        printf("OMP threads  : %d per rank  (%d total)\n",
               num_threads, world_size * num_threads);
        printf("Dict size    : %d\n",  dict_size);
        printf("Top-k        : %d\n",  top_k);
        printf("Metric       : %s\n",  metric_name[metric]);

        /* Read query */
        FILE *qf = fopen(query_file, "r");
        if (!qf || readQuery(qf, dict_size, query_arr) != 0) {
            fprintf(stderr, "Error: Cannot read query file '%s'\n", query_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Read documents */
        FILE *df = fopen(documents_file, "r");
        if (!df) {
            fprintf(stderr, "Error: Cannot open documents file '%s'\n", documents_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        total_docs = getNrOfRows(df);
        if (total_docs <= 0) {
            fprintf(stderr, "Error: Document file is empty or unreadable\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Documents    : %d\n\n", total_docs);

        all_documents = (int *)malloc((size_t)total_docs * dict_size * sizeof(int));
        all_ids       = (int *)malloc((size_t)total_docs * sizeof(int));
        if (!all_documents || !all_ids) {
            fprintf(stderr, "Error: Allocation failed for document arrays\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* readDictionary expects a 2-D VLA pointer – cast the 1-D block */
        int (*docs2d)[dict_size] = (int (*)[dict_size])all_documents;
        if (readDictionary(df, total_docs, dict_size, docs2d, all_ids) != 0) {
            fprintf(stderr, "Error: Failed to parse document file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        t_io_end = MPI_Wtime();
    }

    /* ── Broadcast metadata ── */
    double t_comm_start = MPI_Wtime();

    MPI_Bcast(&total_docs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&metric,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(query_arr, dict_size, MPI_INT, 0, MPI_COMM_WORLD);

    /* ── Compute per-rank document counts (block distribution, ±1 balance) ── */
    int base_count  = total_docs / world_size;
    int remainder   = total_docs % world_size;
    int local_count = base_count + (rank < remainder ? 1 : 0);

    /*
     * Build MPI_Scatterv displacement/count arrays.
     *
     * FIX over original: the original used a manual MPI_Send loop on rank 0,
     * which is O(P) sequential sends.  MPI_Scatterv is a single collective
     * that lets the MPI runtime choose an optimal communication tree.
     */
    int *sendcounts_doc = NULL, *displs_doc = NULL;
    int *sendcounts_id  = NULL, *displs_id  = NULL;

    if (rank == 0) {
        sendcounts_doc = (int *)malloc(world_size * sizeof(int));
        displs_doc     = (int *)malloc(world_size * sizeof(int));
        sendcounts_id  = (int *)malloc(world_size * sizeof(int));
        displs_id      = (int *)malloc(world_size * sizeof(int));

        int off_doc = 0, off_id = 0;
        for (int p = 0; p < world_size; p++) {
            int cnt           = base_count + (p < remainder ? 1 : 0);
            sendcounts_doc[p] = cnt * dict_size;
            displs_doc[p]     = off_doc;
            sendcounts_id[p]  = cnt;
            displs_id[p]      = off_id;
            off_doc          += cnt * dict_size;
            off_id           += cnt;
        }
    }

    int *local_documents = (int *)malloc((size_t)local_count * dict_size * sizeof(int));
    int *local_ids       = (int *)malloc((size_t)local_count * sizeof(int));
    if (!local_documents || !local_ids) {
        fprintf(stderr, "Rank %d: allocation failed for local arrays\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Scatter document feature vectors */
    MPI_Scatterv(all_documents, sendcounts_doc, displs_doc, MPI_INT,
                 local_documents, local_count * dict_size, MPI_INT,
                 0, MPI_COMM_WORLD);

    /* Scatter document IDs */
    MPI_Scatterv(all_ids, sendcounts_id, displs_id, MPI_INT,
                 local_ids, local_count, MPI_INT,
                 0, MPI_COMM_WORLD);

    double t_comm_scatter_end = MPI_Wtime();

    if (rank == 0) {
        free(all_documents); all_documents = NULL;
        free(all_ids);       all_ids       = NULL;
        free(sendcounts_doc); free(displs_doc);
        free(sendcounts_id);  free(displs_id);
    }

    /* ── Parallel similarity computation (OpenMP) + local top-k (min-heap) ──
     *
     * Pattern:
     *   Phase A (parallel)  – each thread computes similarity for its slice
     *   Phase B (sequential)– single-threaded heap insertion
     *
     * We pre-compute all scores into a shared array in Phase A, then build
     * the heap in Phase B.  This avoids any thread-safety concern on the
     * heap while keeping the expensive pow/sqrt calls fully parallelised.
     */
    double t_compute_start = MPI_Wtime();

    double *scores = (double *)malloc((size_t)local_count * sizeof(double));
    if (!scores) { MPI_Abort(MPI_COMM_WORLD, 1); }

    /*
     * Each iteration is independent: no shared writes, no loop-carried
     * dependencies.  schedule(dynamic,64) helps when documents vary in
     * density (e.g. sparse feature vectors with many zeros that exit early).
     */
#pragma omp parallel for schedule(dynamic, 64) default(none) \
    shared(local_documents, query_arr, scores, local_count, dict_size, metric)
    for (int i = 0; i < local_count; i++) {
        scores[i] = computeSimilarity(dict_size,
                                      &local_documents[i * dict_size],
                                      query_arr, metric);
    }

    /* Build min-heap from pre-computed scores (sequential, O(n log k)) */
    int actual_k = (top_k > local_count) ? local_count : top_k;
    MinHeap *heap = createMinHeap(actual_k);
    if (!heap) { MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int i = 0; i < local_count; i++)
        insertHeap(heap, local_ids[i], scores[i]);

    free(scores);

    /* Extract sorted top-k from heap */
    DocScore *local_topk = getTopK(heap);
    destroyMinHeap(heap);

    double t_compute_end = MPI_Wtime();

    /* ── Global top-k reduction ── */
    double t_reduce_start = MPI_Wtime();

    DocScore *global_topk = NULL;
    if (rank == 0)
        global_topk = (DocScore *)malloc(top_k * sizeof(DocScore));

    kreduce(local_topk, actual_k, top_k, world_size, rank, global_topk);

    double t_reduce_end = MPI_Wtime();
    double t_total_end  = MPI_Wtime();

    /* ── Output ── */
    if (rank == 0) {
        printf("=== Top %d Similar Documents ===\n", top_k);
        printf("Rank\tDoc ID\tSimilarity Score\n");
        printf("----\t------\t----------------\n");
        for (int i = 0; i < top_k; i++) {
            if (global_topk[i].id == -1) break;   /* fewer docs than top_k */
            printf("%d\t%d\t%.6f\n", i + 1, global_topk[i].id, global_topk[i].score);
        }

        double t_io      = t_io_end      - t_io_start;
        double t_scatter = t_comm_scatter_end - t_comm_start;
        double t_compute = t_compute_end - t_compute_start;
        double t_reduce  = t_reduce_end  - t_reduce_start;
        double t_total   = t_total_end   - t_total_start;
        double t_comm    = t_scatter + t_reduce;

        printf("\n=== Performance Breakdown ===\n");
        printf("I/O              : %8.6f s  (%5.1f%%)\n", t_io,      100.0*t_io     /t_total);
        printf("Scatter (comm)   : %8.6f s  (%5.1f%%)\n", t_scatter, 100.0*t_scatter/t_total);
        printf("Computation      : %8.6f s  (%5.1f%%)\n", t_compute, 100.0*t_compute/t_total);
        printf("Gather/reduce    : %8.6f s  (%5.1f%%)\n", t_reduce,  100.0*t_reduce /t_total);
        printf("Total comm       : %8.6f s  (%5.1f%%)\n", t_comm,    100.0*t_comm   /t_total);
        printf("Total            : %8.6f s\n", t_total);
        printf("Compute fraction : %5.1f%%  (parallel efficiency indicator)\n",
               100.0 * t_compute / t_total);
        printf("Theoretical max speedup (Amdahl, P=%d): %.2fx\n",
               world_size * num_threads,
               1.0 / ((1.0 - t_compute/t_total) + (t_compute/t_total)/(world_size*num_threads)));

        free(global_topk);
    }

    free(local_documents);
    free(local_ids);
    free(local_topk);
    free(query_arr);

    MPI_Finalize();
    return 0;
}
