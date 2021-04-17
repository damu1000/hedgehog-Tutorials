// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of 
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.

//g++-8 -I/home/damodars/hedgehog/hedgehog/build/include/ ./tutorial0.cc -std=c++17 -pthread

#include <hedgehog/hedgehog.h>

//wrappers for input and output
struct inputVec{
	int * in, *out;
	size_t size;
	inputVec(int *i, int *o, size_t s) : in(i), out(o), size(s) {}
};

struct outputVec{
	int * out;
	outputVec(int *o) : out(o) {}
};

//vector add task
class vecAddTask : public hh::AbstractTask<outputVec, inputVec> {
 public:
	vecAddTask(std::string_view const &name, size_t numberThreads)
      : hh::AbstractTask<outputVec, inputVec>(name, numberThreads) {}

  void execute(std::shared_ptr<inputVec> inputVec) override {

    for (size_t i = 0; i < inputVec->size; ++i)
    	inputVec->out[i] += inputVec->in[i];

    auto result = std::make_shared<outputVec>(inputVec->out);
    this->addResult(result);
  }

  std::shared_ptr<hh::AbstractTask<outputVec, inputVec>> copy() override {
    return std::make_shared<vecAddTask>(this->name(), this->numberThreads());
  }
};

int main(int argc, char **argv) {

  size_t numberThread = 3, len = 10, blockSize = 2, numberBlocks = len/blockSize;

  //allocate vectors
  int *dataA = nullptr, *dataB = nullptr;
  dataA = new int[len]();
  dataB = new int[len]();
  for(size_t i = 0; i < len; ++i){
      dataA[i] = i;
      std::cout << dataA[i] << " ";
  }
  std::cout << std::endl;
  for(size_t i = 0; i < len; ++i){
	  dataB[i] = i;
	  std::cout << dataB[i] << " ";
  }
  std::cout << std::endl;

  // Declaring and instantiating the graph
  hh::Graph<outputVec, inputVec> graph("Tutorial 0 : Vector Add");

  // Declaring and instantiating the task
  auto task = std::make_shared<vecAddTask>("Tutorial 0", numberThread);

  // Set the task as the task that will be connected to the graph input
  graph.input(task);

  // Set the task as the task that will be connected to the graph output
  graph.output(task);

  // Execute the graph
  graph.executeGraph();

  // Create blocks and send them to the graph
  for (size_t i = 0; i < numberBlocks; ++i) {
      auto input = std::make_shared<inputVec>(&dataA[i*blockSize], &dataB[i*blockSize], blockSize);
      graph.pushData(input);
  }

  // Notify the graph that no more data will be sent
  graph.finishPushingData();

  // Wait for everything to be processed
  graph.waitForTermination();

  //Print the result matrix
  for(size_t i = 0; i < len; ++i){
	  std::cout << dataB[i] << " ";
  }
  std::cout << std::endl;


  delete[] dataA;
  delete[] dataB;

}
