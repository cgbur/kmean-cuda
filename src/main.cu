#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <chrono>

struct Point {
  float x{}, y{};
};


void assign_clusters_gpu(Point *dev_points, int *cluster, Point *centers, unsigned int n, unsigned int k);

void assign_clusters(Point *points, int *cluster, Point *centers, unsigned int n, unsigned int k,
                     float *dists);

void sum_clusters(const Point *points, const int *cluster, float *x_sum, float *y_sum, int *count, unsigned int n);

void update_clusters(Point *centers, unsigned int k, const float *x_sum, const float *y_sum,
                     const int *count);

void setup(Point *points,
           Point *centers,
           unsigned int n,
           unsigned int k);

bool has_converged(unsigned int k,
                   const int *count,
                   int *old_count);

void dump_to_csv(int iter, Point *data, int *cluster, unsigned int count);

void reset(int *arr, unsigned int count) {
  for (int i = 0; i < count; ++i) {
    arr[i] = 0;
  }
}

void reset(float *arr, unsigned int count) {
  for (int i = 0; i < count; ++i) {
    arr[i] = 0.0f;
  }
}

bool verbose = false;
bool dump = false;

unsigned int big_n = 1e5;
unsigned int threads_per = 512;

void run(unsigned int n, unsigned int k, unsigned int threads, bool gpu);

int main() {
  printf("n,k,num_iterations,mflops,gpu,num_threads\n");
  for (int num = 1000; num < 1e8; num *= 10)
    for (int k = 50; k < 1000; k += 50){
      run(num, k, threads_per, true);
      run(num, k, threads_per, false);
    }
}

void run(unsigned int n, unsigned int k, unsigned int threads, bool gpu) {
  big_n = n;
  threads_per = threads;
  unsigned int t = 30; // iterations
  auto *points = new Point[n];
  auto *centers = new Point[k];
  auto *clusters = new int[n];


  if (verbose) {
    printf("n = %i (num points)\n", n);
    printf("t = %i (max num iterations)\n", t);
    printf("k = %i (num clusters)\n", k);
  }

  setup(points, centers, n, k);

  /**
   * Points don't change so lets add them to the GPU
   */
  Point *device_points;
  cudaMalloc(&device_points, sizeof(Point) * n);
  cudaMemcpy(device_points, points, sizeof(Point) * n, cudaMemcpyHostToDevice);

  auto dists = new float[k];
  auto x_sum = new float[k];
  auto y_sum = new float[k];
  auto count = new int[k];
  auto old_count = new int[k];

  int num_iterations = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < t; ++i) {

    if (gpu) {
      assign_clusters_gpu(device_points, clusters, centers, n, k);
    } else {
      assign_clusters(points, clusters, centers, n, k, dists);
    }

    reset(x_sum, k);
    reset(y_sum, k);
    reset(count, k);

    sum_clusters(points, clusters, x_sum, y_sum, count, n);
    update_clusters(centers, k, x_sum, y_sum, count);

    if (dump)
      dump_to_csv(i, points, clusters, n);

    num_iterations++;

    if (has_converged(k, count, old_count))
      break;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if (verbose) {
    printf("\nCenters:\n  (nan centers were not mapped)\n"
           " cluster_id: (x, y) num_points\n");
    for (int i = 0; i < k; ++i) {
      printf("  %i: (%f, %f) w/ %i points\n", i, centers[i].x, centers[i].y, count[i]);
    }

    printf("\nFinished using %i iterations\n", num_iterations);
  }
  auto interactions = ((unsigned int) k * (unsigned int) n * (unsigned int) num_iterations);

  printf("%i,%i,%i,%f,%i,%i\n", n, k, num_iterations, (float) interactions / (float) duration, gpu, threads);

  cudaFree(device_points);
}

bool has_converged(unsigned int k, const int *count, int *old_count) {
  bool should_end = true;
  for (int j = 0; j < k; ++j) {
    if (count[j] != old_count[j]) {
      should_end = false;
      old_count[j] = count[j];
    }
  }
  return should_end;
}

void setup(Point *points, Point *centers, unsigned int n, unsigned int k) {
  std::random_device rd;
  std::mt19937 mt(5);
  std::uniform_real_distribution<float> xy_dist(0.0, 60.0);
  std::uniform_real_distribution<float> std_dist(1.25, 7.5);

  unsigned int num_per_k = n / k;
  for (int i = 0; i < k; ++i) {
    float x = xy_dist(mt);
    float y = xy_dist(mt);
    float sx = std_dist(mt);
    float sy = std_dist(mt);
    std::normal_distribution<float> norm_x(x, sx);
    std::normal_distribution<float> norm_y(y, sy);

    for (int j = 0; j < num_per_k; ++j) {
      points[i * num_per_k + j].x = norm_x(mt);
      points[i * num_per_k + j].y = norm_y(mt);
    }

    centers[i].x = xy_dist(mt);
    centers[i].y = xy_dist(mt);
  }
}

void update_clusters(Point *centers, unsigned int k, const float *x_sum, const float *y_sum,
                     const int *count) {
  for (int c_idx = 0; c_idx < k; c_idx++) {
    centers[c_idx].x = x_sum[c_idx] / (float) count[c_idx];
    centers[c_idx].y = y_sum[c_idx] / (float) count[c_idx];
  }
}

void sum_clusters(const Point *points, const int *cluster, float *x_sum, float *y_sum, int *count, unsigned int n) {
  for (int i = 0; i < n; ++i) {
    x_sum[cluster[i]] += points[i].x;
    y_sum[cluster[i]] += points[i].y;
    count[cluster[i]] += 1;
  }
}

void assign_clusters(Point *points, int *cluster, Point *centers, unsigned int n, unsigned int k,
                     float *dists) {
  for (int i = 0; i < n; ++i) {
    for (int cluster_idx = 0; cluster_idx < k; ++cluster_idx) {
      float x = points[i].x - centers[cluster_idx].x;
      float y = points[i].y - centers[cluster_idx].y;
      dists[cluster_idx] = sqrt(pow(x, 2) + pow(y, 2));
    }
    // https://riptutorial.com/cplusplus/example/11151/find-max-and-min-element-and-respective-index-in-a-vector
//    cluster[i] = std::min_element(dists, dists + k) - dists;

    float min = dists[0];
    int min_idx = 0;
    for (int j = 1; j < k; ++j) {
      if (min > dists[j]) {
        min = dists[j];
        min_idx = j;
      }
    }
    cluster[i] = min_idx;
  }
}

__global__ void assign_clusters_kernel(Point *points,
                                       int *clusters,
                                       Point *centers,
                                       float *dists,
                                       unsigned int n,
                                       unsigned int k) {
//  extern __shared__ float dists[];
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;

//  int dists_idx = 0;
  int dists_idx = i * k;
  for (int cluster_idx = 0; cluster_idx < k; ++cluster_idx) {
    float x = points[i].x - centers[cluster_idx].x;
    float y = points[i].y - centers[cluster_idx].y;
    dists[dists_idx + cluster_idx] = sqrt(pow(x, 2) + pow(y, 2));
  }
  // https://riptutorial.com/cplusplus/example/11151/find-max-and-min-element-and-respective-index-in-a-vector
//  clusters[i] = std::min_element(dists, dists + k) - dists;
  // find min index

//  __syncthreads();
  float min = dists[dists_idx];
  int min_idx = dists_idx;
  for (int j = 1; j < k; ++j) {
    if (min > dists[dists_idx + j]) {
      min = dists[dists_idx + j];
      min_idx = dists_idx + j;
    }
  }
  clusters[i] = min_idx - dists_idx;
}

void assign_clusters_gpu(Point *dev_points, int *cluster, Point *centers, unsigned int n, unsigned int k) {
  Point *dev_centers; // centers pointer
  cudaMalloc(&dev_centers, sizeof(Point) * k);
  cudaMemcpy(dev_centers, centers, sizeof(Point) * k, cudaMemcpyHostToDevice);

  int *dev_cluster; // make cluster array but dont fill
  cudaMalloc(&dev_cluster, sizeof(int) * n);

  float *dev_dists; // make cluster array but dont fill
  cudaMalloc(&dev_dists, sizeof(float) * k * n);
//  auto *dists = new float[k * blockDim.x];


  int num_blocks = std::ceil(big_n / threads_per);
  assign_clusters_kernel<<<num_blocks, threads_per>>>(dev_points,
                                                      dev_cluster,
                                                      dev_centers,
                                                      dev_dists,
                                                      n, k);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaMemcpy(cluster, dev_cluster, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dev_centers);
  cudaFree(dev_cluster);
  cudaFree(dev_dists);
}

void dump_to_csv(int iter, Point *data, int *cluster, unsigned int count) {
  std::ofstream myFile(R"(B:\code\633\final-py\iters\out)" + std::to_string(iter) + ".csv");

  myFile << "x,y,cluster\n";
  for (int i = 0; i < count; ++i) {
    myFile << data[i].x;
    myFile << ",";
    myFile << data[i].y;
    myFile << ",";
    myFile << cluster[i];
    myFile << "\n";

  }
  // Close the file
  myFile.close();
}