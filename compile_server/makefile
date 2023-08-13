compile_server:compile_server.cc
	g++ -o $@ $^ -I /usr/include/python3.6m -L/usr/lib/python3.6 -lpython3.6m -std=c++11 -ljsoncpp -lpthread
.PHONY:clean
clean:
	rm -f compile_server