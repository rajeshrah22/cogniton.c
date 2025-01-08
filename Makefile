CC = gcc
CFLAGS = -Wextra -Wall -Werror -Wpedantic

default: test

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

test: tensor.o test.o
	$(CC) $(CFLAGS) -o $@ $^

linear-reg: tensor.o linear-reg.o
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f *.o test linear-reg
