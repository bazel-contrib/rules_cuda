#include <thrust/device_vector.h>

#include <iostream>

int main() {
  const int num_elements = 8192;
  thrust::device_vector<float> vec(num_elements, 42.0);
  auto sum = thrust::reduce(vec.begin(), vec.end(), (float)0.0, thrust::plus<float>());
  std::cout << "thrust device_vector created, sum reduce as " << sum << ", mean: " << sum / num_elements << std::endl;
  return 0;
}
