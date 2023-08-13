#include "compile_run.hpp"
#include "../comm/httplib.h"
#include "fs_nas.hpp"
#include <cstdlib>
using namespace ns_compile_and_run;
using namespace httplib;

void Usage(std::string proc)
{
    std::cerr << "Usage: " << "\n\t" << proc << " port" << std::endl;
}

//编译服务随时可能被多个人请求，必须保证传递上来的code，形成源文件名称的时候，要具有
//唯一性，要不然多个用户之间会互相影响
//./compile_server port
int main(int argc, char *argv[])
{
    if(argc != 2){
        Usage(argv[0]);
        return 1;
    }

    Server svr;

    svr.Post("/compile_and_run", [](const Request &req, Response &resp){
        std::cout << "进入了数据结构" << std::endl;
        // 用户请求的服务正文是我们想要的json string
        std::string in_json = req.body;
        std::string out_json;
        if(!in_json.empty()){
            CompileAndRun::Start(in_json, &out_json);
            resp.set_content(out_json, "application/json;charset=utf-8");
        }
    });
    svr.Post("/select_and_nas", [](const Request &req, Response &resp){

        std::cout << "进入了特征选择" << std::endl;
        // 用户请求的服务正文是我们想要的json string
        std::string in_json = req.body;
        std::string out_json;
        Json::Value in_value;
        Json::Reader reader;
        reader.parse(in_json, in_value); //最后在处理差错问题
        int number = std::atoi(in_value["number"].asCString());
        start(1,&out_json);
        std::cout << out_json << std::endl;
        resp.set_content(out_json, "application/json;charset=utf-8");
        
    });
    // svr.set_base_dir("./wwwroot");
    svr.listen("0.0.0.0", atoi(argv[1])); //启动http服务

    return 0;
}