#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <float.h>   /* DBL_MAX for sentinel padding */

/* Configuration constants */
#define MAX_LINE_LENGTH 8192
#define DEFAULT_K 10

/* Similarity metric types */
typedef enum {
    SIMILARITY_POWER,
    SIMILARITY_COSINE,
    SIMILARITY_EUCLIDEAN,
    SIMILARITY_MANHATTAN
} SimilarityMetric;

/* Document ID + score pair */
typedef struct {
    int    id;
    double score;
} DocScore;

/* Min-heap for O(n log k) top-k selection */
typedef struct {
    DocScore *heap;
    int       capacity;   /* == k */
    int       size;
} MinHeap;

/* ── File I/O ────────────────────────────────── */
int getNrOfRows(FILE *file);
int readDictionary(FILE *file, int rows, int cols,
                   int documents[][cols], int ids[]);
int readQuery(FILE *file, int elements, int queryArr[]);

/* ── Similarity ──────────────────────────────── */
double computeSimilarity(int cols, int document[], int query[],
                         SimilarityMetric metric);
double powerSimilarity    (int cols, int document[], int query[]);
double cosineSimilarity   (int cols, int document[], int query[]);
double euclideanDistance  (int cols, int document[], int query[]);
double manhattanDistance  (int cols, int document[], int query[]);

/* ── Min-heap ────────────────────────────────── */
MinHeap *createMinHeap (int k);
void     destroyMinHeap(MinHeap *heap);
void     insertHeap    (MinHeap *heap, int id, double score);
void     heapifyDown   (MinHeap *heap, int idx);
void     heapifyUp     (MinHeap *heap, int idx);
DocScore *getTopK      (MinHeap *heap);

/* ── Sorting ─────────────────────────────────── */
void mergeSortDocScores(DocScore arr[], int left, int right);
void mergeDocScores    (DocScore arr[], int left, int mid, int right);

/* ── Misc ────────────────────────────────────── */
void   printUsage  (const char *progName);
double getWallTime (void);

#endif /* UTILS_H */
