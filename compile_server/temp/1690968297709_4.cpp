#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution
{
public:
    int Max(const vector<int> &v)
    {
        //将你的代码写在下面
        int ret = -99999;
        for (auto i : v) {
            /* code */
            if(ret < i){
                ret = i;
            }
        }

        return ret;
    }
};

#ifndef COMPILER_ONLINE
#include "header.cpp"
#endif

void Test1()
{
    vector<int> v = {1, 2, 3, 4, 5, 6};
    int max = Solution().Max(v);
    if (max == 6)
    {
        std::cout << "Test 1 .... OK" << std::endl;
    }
    else
    {
        std::cout << "Test 1 .... Failed" << std::endl;
    }
}

void Test2()
{
    vector<int> v = {-1, -2, -3, -4, -5, -6};
    int max = Solution().Max(v);
    if (max == -1)
    {
        std::cout << "Test 2 .... OK" << std::endl;
    }
    else
    {
        std::cout << "Test 2 .... Failed" << std::endl;
    }
}

int main()
{
    Test1();
    Test2();

    return 0;
}
