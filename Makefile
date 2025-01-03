CC = gcc
CFLAGS = -Wextra -Wall -Werror -Wpedantic

test_deps = tensor.o test.o

default: test

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

test: $(test_deps)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f *.o test
