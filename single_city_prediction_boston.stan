
data {
  int<lower=1> T;     // num observations
   real y[T];         // observed outputs
   int<lower=0> n_new;// Predictions
}

transformed data{
   int<lower=0> x=T+n_new;
}

parameters {
real mu; // mean coeff
real phi; // autoregression coeff
real theta; // moving avg coeff
real<lower=0> sigma; // noise scale
}

transformed parameters{
  vector[T] nu; // prediction for time t
  vector[T] err; // error for time t
  
  nu[1] = mu + phi * mu; // assume err[0] == 0
  err[1] = y[1] - nu[1];   // first error term
  for (t in 2:T) {
     nu[t] = mu + phi*y[t-1] + theta*err[t-1];
     err[t] = y[t] - nu[t];
  }
}

model {
  mu ~ normal(0, 10); // priors
  phi ~ normal(0, 100);
  theta ~ normal(0, 100);
  sigma ~ normal(0, 100);
  err ~ normal(0, sigma); // likelihood
}

generated quantities{
   vector[x] predict;
   vector[x] nu_predict;
   vector[x] err_predict;
   
   for(i in 1:x){
     if(i <= T){
       predict[i] = normal_rng(nu[i],sigma);
       nu_predict[i] = nu[i];
       err_predict[i] = err[i];
     } 
     else{
       nu_predict[i] = mu + phi*predict[i-1] + theta*err_predict[i-1];
       predict[i] = normal_rng(nu_predict[i],sigma);
       err_predict[i] = predict[i]-nu_predict[i];
     }
  }
}

