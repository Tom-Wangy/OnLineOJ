#include <Python.h>
#include <iostream>
#include <thread>
#include <jsoncpp/json/json.h>
#include<string>
using namespace std;
 

int start(int number, std::string *out_json){
    std::cout << "现在进入start函数了" << endl;
    // 1、初始化python接口  
	Py_Initialize();
	if(!Py_IsInitialized()){
		cout << "python init fail" << endl;
		return -1;
	}
    // 2、初始化python系统文件路径，保证可以访问到 .py文件
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('../select_cf')");
    PyObject* sysPath = PySys_GetObject("path");
    PyObject* currentDir = PyUnicode_DecodeFSDefault(".");
    PyList_Append(sysPath, currentDir);
    // PySys_SetPath("/root/C++_program/OnlineOJ/select_cf");
 
    // 3、调用python文件名，不用写后缀
    // PyImport_ImportModule("hyperparameter");
	PyObject* pModule = PyImport_ImportModule("test1");

	if( pModule == NULL ){
		cout <<"module not found" << endl;
		return -1;
	}
    // 4、调用函数
	PyObject* pFunc = PyObject_GetAttrString(pModule, "run");
	if( !pFunc || !PyCallable_Check(pFunc)){
		cout <<"not found function add_num" << endl;
		return -1;
	}

    // 创建参数并传递给Python函数
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyLong_FromLong(number));
    // 
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    // 检查返回值是否为字典类型
    if (!PyDict_Check(pResult)) {
        std::cout << "Result is not a dictionary" << std::endl;
        return -1;
    }


    Json::Value out_value;
    // 获取字典中的值
    PyObject* pKey1 = PyUnicode_FromString("std_out");
    PyObject* pValue1 = PyDict_GetItem(pResult, pKey1);
    Py_DECREF(pKey1);

    PyObject* pKey2 = PyUnicode_FromString("std_err");
    PyObject* pValue2 = PyDict_GetItem(pResult, pKey2);
    Py_DECREF(pKey2);

    if (pValue1 == NULL || pValue2 == NULL ) {
        std::cout << "Key not found in dictionary" << std::endl;
        return -1;
    }

    // 将获取的值转换为字符串并存入json
    char* _stdout = PyUnicode_AsUTF8(pValue1);
    char* _stderr = PyUnicode_AsUTF8(pValue2);
    out_value["stdout"] = _stdout;
    out_value["stderr"] = _stderr;
    // std::cout << out_value["stdout"] << out_value["stderr"] << std::endl;

    //将json转化为字符串
    Json::StyledWriter writer;
    *out_json = writer.write(out_value);
    Py_DECREF(pArgs);
    // 5、结束python接口初始化
	Py_Finalize();

    return 0;
}


