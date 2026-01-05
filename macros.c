#include<stdio.h>


#define PI 3.14
#define AREA(r) (PI * r * r)

#ifdef radius
#define radius 7
#endif

#if radius >10
#define radius 10
#elif radius < 5
#define radius 5
#else 
#define radius 7
#endif

int main(){
    printf("Radius: %d , Area : %2f\n", radius, AREA(radius));
}
