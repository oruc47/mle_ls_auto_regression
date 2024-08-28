/*
Code written by Ikbal Oruc
moruc@ku.edu.tr
*/



#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <numeric>

#include "func.h"



using namespace std;

int main(){
    solve_mle();
    solve_ar();
    monte_carlo();
    simulation(200, 500);
    simulation(10, 500);
    simulation(80,500);
    simulation(320, 500);    
}   
