# include<iostream>
using namespace std;

int main(){
    vector<int> wights = {1, 3, 4};
    vector<int> values = {15, 20, 30};
    int bag = 4;
    vector<int> dp(bag+1, 0);
    for(int i=0; i<weights.size(); i++){
        for(int j=bag; j>weights[i]; j--){
            dp[j] = max(dp[j], dp[j - weigths[i]] + values[i]);
        }
    }
    return dp[bag];
}
