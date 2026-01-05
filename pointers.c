#include<stdio.h>

// int main(){
//     int x = 10;
//     int* ptr1 = &x;
//     printf("Address of x: %p\n" , ptr1);
//     printf("Address of x: %d\n" , *ptr1);

// }


// int main(){
//     int num = 10;
//     float fnum = 3.14;
//     void* vptr;

//     vptr =&num;
//     printf("integer: %d\n" , *(int*)vptr);

//     vptr = &fnum;
//     printf("Float:%.2f\n", *(float*)vptr);

// }

#include<stdlib.h>
// int main(){
//     int* ptr = NULL;
//     printf("1. Initial ptr value: %p\n", (void*)ptr);

//     if(ptr == NULL){
//         printf("2. ptr is NULL, cannont dereference\n");
//     }

//     ptr = malloc(sizeof(int));
//     if (ptr == NULL){
//         printf("3. Memory allocation failed\n");
//         return 1;
//     }
// }

// int main(){
//     int arr[] = {12,12,23,34,35};
//     int* ptr = arr;

//     for (int i = 0; i<5; i++){
//         printf("%d ", *ptr);
//         printf("%p\n", *ptr);
//         ptr++;
//     }
// }


int main(){
    int arr1[] = {1,2,3,4,5};
    int arr2[] = {6,7,8,9,10};
    int* ptr1 = arr1;
    int*ptr2 = arr2;

    int* matrix[] = {ptr1 , ptr2};

    for (int i = 0; i<2; i++){
        for(int j =0 ; j<4; j++){
            printf("%d ", *matrix[i]++);
        }
        printf("\n");
    }
}