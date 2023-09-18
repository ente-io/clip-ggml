#include <string>

extern "C"
{
  char * test(char * input) {
    // std::string sample = "test works";
    // char *result = new char[sample.size() + 1];
    // strcpy(result, sample.c_str());
    // return result;
    return input;
  }
}
