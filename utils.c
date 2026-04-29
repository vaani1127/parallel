#include "utils.h"
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include <float.h>

/* ═══════════════════════════════════════════════════════════════
 * File I/O
 * ═══════════════════════════════════════════════════════════════ */

int getNrOfRows(FILE *file) {
    if (!file) return -1;
    int rowCount = 0;
    char buffer[MAX_LINE_LENGTH];
    while (fgets(buffer, sizeof(buffer), file))
        rowCount++;
    rewind(file);
    return rowCount;
}

int readDictionary(FILE *file, int rows, int cols,
                   int documents[][cols], int ids[]) {
    if (!file || !documents || !ids) return -1;
    char buffer[MAX_LINE_LENGTH];

    for (int i = 0; i < rows; i++) {
        if (!fgets(buffer, sizeof(buffer), file)) {
            fprintf(stderr, "Error: Unexpected end of file at row %d\n", i);
            return -1;
        }
        char *token = strtok(buffer, ":");
        if (!token) {
            fprintf(stderr, "Error: Invalid format at row %d\n", i);
            return -1;
        }
        ids[i] = atoi(token);

        for (int j = 0; j < cols; j++) {
            token = strtok(NULL, " \t\n");
            if (!token) {
                fprintf(stderr, "Error: Insufficient columns at row %d, col %d\n", i, j);
                return -1;
            }
            documents[i][j] = atoi(token);
        }
    }
    fclose(file);
    return 0;
}

int readQuery(FILE *file, int elements, int queryArr[]) {
    if (!file || !queryArr) return -1;
    char buffer[MAX_LINE_LENGTH];

    if (!fgets(buffer, sizeof(buffer), file)) {
        fprintf(stderr, "Error: Empty query file\n");
        return -1;
    }

    char *token = strtok(buffer, " \t\n");
    for (int i = 0; i < elements; i++) {
        if (!token) {
            fprintf(stderr, "Error: Insufficient query elements (need %d)\n", elements);
            return -1;
        }
        queryArr[i] = atoi(token);
        token = strtok(NULL, " \t\n");
    }
    fclose(file);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Similarity metrics
 * ═══════════════════════════════════════════════════════════════ */

double computeSimilarity(int cols, int document[], int query[],
                         SimilarityMetric metric) {
    switch (metric) {
        case SIMILARITY_COSINE:    return cosineSimilarity (cols, document, query);
        case SIMILARITY_EUCLIDEAN: return euclideanDistance(cols, document, query);
        case SIMILARITY_MANHATTAN: return manhattanDistance(cols, document, query);
        default:                   return powerSimilarity  (cols, document, query);
    }
}

/* Power-based: Σ doc[i]^query[i]  (original metric) */
double powerSimilarity(int cols, int document[], int query[]) {
    double result = 0.0;
    for (int i = 0; i < cols; i++)
        result += pow((double)document[i], (double)query[i]);
    return result;
}

/* Cosine similarity: dot(A,B) / (||A|| · ||B||) ∈ [-1, 1] */
double cosineSimilarity(int cols, int document[], int query[]) {
    double dot = 0.0, normD = 0.0, normQ = 0.0;
    for (int i = 0; i < cols; i++) {
        dot   += (double)document[i] * (double)query[i];
        normD += (double)document[i] * (double)document[i];
        normQ += (double)query[i]    * (double)query[i];
    }
    if (normD == 0.0 || normQ == 0.0) return 0.0;
    return dot / (sqrt(normD) * sqrt(normQ));
}

/* Euclidean distance (negated so higher = more similar) */
double euclideanDistance(int cols, int document[], int query[]) {
    double sum = 0.0;
    for (int i = 0; i < cols; i++) {
        double diff = (double)document[i] - (double)query[i];
        sum += diff * diff;
    }
    return -sqrt(sum);
}

/*
 * Manhattan distance (negated so higher = more similar).
 *
 * BUG FIX: the original code used abs() on a double expression,
 * which silently truncates to int before taking absolute value.
 * The correct function for double is fabs() from <math.h>.
 */
double manhattanDistance(int cols, int document[], int query[]) {
    double sum = 0.0;
    for (int i = 0; i < cols; i++)
        sum += fabs((double)document[i] - (double)query[i]);  /* fabs, not abs */
    return -sum;
}

/* ═══════════════════════════════════════════════════════════════
 * Min-heap (top-k selection in O(n log k))
 * ═══════════════════════════════════════════════════════════════ */

MinHeap *createMinHeap(int k) {
    MinHeap *h = (MinHeap *)malloc(sizeof(MinHeap));
    if (!h) return NULL;
    h->capacity = k;
    h->size     = 0;
    h->heap     = (DocScore *)malloc(k * sizeof(DocScore));
    if (!h->heap) { free(h); return NULL; }
    return h;
}

void destroyMinHeap(MinHeap *heap) {
    if (heap) {
        free(heap->heap);
        free(heap);
    }
}

void heapifyDown(MinHeap *heap, int idx) {
    int smallest = idx;
    int left  = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left  < heap->size && heap->heap[left ].score < heap->heap[smallest].score)
        smallest = left;
    if (right < heap->size && heap->heap[right].score < heap->heap[smallest].score)
        smallest = right;

    if (smallest != idx) {
        DocScore tmp        = heap->heap[idx];
        heap->heap[idx]     = heap->heap[smallest];
        heap->heap[smallest]= tmp;
        heapifyDown(heap, smallest);
    }
}

void heapifyUp(MinHeap *heap, int idx) {
    if (idx == 0) return;
    int parent = (idx - 1) / 2;
    if (heap->heap[idx].score < heap->heap[parent].score) {
        DocScore tmp          = heap->heap[idx];
        heap->heap[idx]       = heap->heap[parent];
        heap->heap[parent]    = tmp;
        heapifyUp(heap, parent);
    }
}

void insertHeap(MinHeap *heap, int id, double score) {
    if (!heap) return;
    if (heap->size < heap->capacity) {
        heap->heap[heap->size].id    = id;
        heap->heap[heap->size].score = score;
        heap->size++;
        heapifyUp(heap, heap->size - 1);
    } else if (score > heap->heap[0].score) {
        heap->heap[0].id    = id;
        heap->heap[0].score = score;
        heapifyDown(heap, 0);
    }
}

/* Returns a malloc'd array sorted descending by score (caller must free). */
DocScore *getTopK(MinHeap *heap) {
    if (!heap || heap->size == 0) return NULL;
    DocScore *result = (DocScore *)malloc(heap->size * sizeof(DocScore));
    if (!result) return NULL;
    memcpy(result, heap->heap, heap->size * sizeof(DocScore));
    mergeSortDocScores(result, 0, heap->size - 1);
    return result;
}

/* ═══════════════════════════════════════════════════════════════
 * Merge sort (descending by score)
 * ═══════════════════════════════════════════════════════════════ */

void mergeDocScores(DocScore arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    DocScore *L = (DocScore *)malloc(n1 * sizeof(DocScore));
    DocScore *R = (DocScore *)malloc(n2 * sizeof(DocScore));
    if (!L || !R) { free(L); free(R); return; }

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid  + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
        arr[k++] = (L[i].score >= R[j].score) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L); free(R);
}

void mergeSortDocScores(DocScore arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortDocScores(arr, left,    mid);
        mergeSortDocScores(arr, mid + 1, right);
        mergeDocScores    (arr, left,    mid, right);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Misc
 * ═══════════════════════════════════════════════════════════════ */

void printUsage(const char *progName) {
    printf("Usage: mpirun -np <P> %s <dict_size> <top_k> "
           "<docs_file> <query_file> [metric] [threads]\n", progName);
    printf("  metric  : power | cosine | euclidean | manhattan  (default: power)\n");
    printf("  threads : OpenMP threads per MPI rank             (default: 1)\n");
    printf("\nExample:\n");
    printf("  mpirun -np 4 %s 100 10 documents.txt query.txt cosine 2\n", progName);
}

double getWallTime(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}
