// #include<stdio.h>
// #include<stdlib.h>

// int main(){
//     int arr[] = {12,14,15,6,7};

//     size_t size = sizeof(arr) / sizeof(arr[0]);
//     printf("Size of arr: %zu\n", size);
//     printf("size of size_t: %zu\n", sizeof(size_t));
//     printf("int size in bytes: %zu\n", sizeof(int));
// }

// #include <stdio.h>
// #include <stddef.h>  // Required for size_t
// #include <string.h>  // Required for strlen

// int main() {
//     char my_string[] = "Hello, world!";
//     size_t length = strlen(my_string);
//     size_t size_of_int = sizeof(int);

//     // Use %zu format specifier for size_t in printf (C99 standard)
//     printf("Length of string: %zu\n", length);
//     printf("Size of int on this system: %zu bytes\n", size_of_int);

//     // Using size_t for a loop counter
//     for (size_t i = 0; i < length; ++i) {
//         printf("%c", my_string[i]);
//     }
//     printf("\n");

//     return 0;
// }


#include<stdio.h>
typedef struct {
    float x;
    float y;
} Point;

int main(){ 
    Point p = {1.1, 2.5};
    printf("size of Point: %zu\n" , sizeof(Point));
}
