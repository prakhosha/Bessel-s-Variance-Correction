<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
        }
    });
    </script>
      
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

***

# Bessel's Variance Correction

**Why We Use $n−1$ to Estimate Variance**

*Posted on July 2023*

***

Statistics is all about estimating parameters of processes that produce random variables. An example of a parameter would be variance which describe or represent the variability of data from a process. Often, we use this equation to calculate variance

$$\hat{\sigma}^2 = \frac{1}{n-1}\sum_{i} ^{}(Y_i - \bar{Y})^2$$

where $Y_i$ is a random variable from a process and $\bar{Y}$ is the estimated sample mean.

But why do we use $n - 1$ instead of $n$ in variance while we use $n$ in mean? The answer lies in estimation theory.

## Bias of Estimators

I have talked a little bit about biasedness of estimator [here]("https://prakhosha.github.io/Jensen-s-Inequality-and-The-Biasedness-of-Standard-Deviation/"). Statistics is trying to model real world processes. Because it is a model, it has parameters. Parameters can be obtained by applying transformation to the data of the process. All transformations of random variables result in random variables. And because it is a random variable, it has a probability distribution. This means that our model parameters, which we used to describe the model of the process, can be different than the true process.

When the central tendency of the parameter of the model different from the true model, we called this bias or mathematically

$$E(\hat{\theta}) \neq \theta$$

where $\hat{\theta}$ is our estimator and $\theta$ is the true parameter.

The variance $\sigma^2$ of a random variable $Y_i$ can be described as

$$\sigma^2 = E(Y_i-\mu)^2$$

If we have a process that produce data consists of $1, 2, 3, 4, 5$ we would estimate that this process has mean of $3$. And so unconsciously, we would compute the variance of this process as

$$\frac{1}{5}(1-3)^2 + \frac{1}{5}(2-3)^2 + \frac{1}{5}(3-3)^2 + \frac{1}{5}(4-3)^2 + \frac{1}{5}(5-3)^2$$

and this result in variance of $2$. But most people would divide the summation by $n-1$ instead of $n$ and this give a variance of $2.5$.

But why would we use $n-1$ instead of $n$? We use $n$ in mean so it is natural we also use $n$ in variance, why would we need $n-1$?

The answer is that this estimator of variance is bias and in order to correct this bias we need some correction. This so called correction is Bessel's correction.

## Bessel's correction

How do we remove this bias? Obviously, to remove the bias we should know how much is the bias first. The true variance of the process is given by

$$\sigma^2 = E(Y_i-\mu)^2$$

What we are computing earlier is actually

$$E(Y_i-\bar{Y})$$

Why? because we use the mean of the data $\bar{Y}$ instead of the true mean $\mu$. But we can not help but use $\bar{Y}$ because we do not know and will never know about the true mean $\mu$. Now if we add and substract $\mu$ from this equation we get

$$E(Y_i-\bar{Y})^2 = E\{(Y_i-\mu) - (\bar{Y}-\mu)\}^2$$

If we expand it,

 $$E\{(Y_i-\mu)^2 + (\bar{Y_i}-\mu)^2 - 2(Y_i-\mu)(\bar{Y}-\mu)\}$$

By linearity and additivity of expectation we get

$$E\{(Y_i-\mu)^2\} + E\{(\bar{Y}-\mu)^2\} - 2E\{(Y_i-\mu)(\bar{Y}-\mu)\}$$

The left side of the equation is the variance of $Y_i$ and its value is $\sigma^2$. The middle part of the equation is variance of the mean and it is $\sigma^2/n$. Now, for the right term we get

 $$E\{(Y_i-\mu)(\bar{Y}-\mu)\}$$

If we expand the $\bar{Y}$ we get

 $$E\{(Y_i - \mu)[\frac{1}{n}(Y_1 - \mu) + \frac{1}{n}(Y_2 - \mu) + ... + \frac{1}{n}(Y_n - \mu)]\}$$

By linearity and additivity of expectation we get

$$\frac{1}{n}E\{(Y_i - \mu)(Y_1 - \mu)\} + \frac{1}{n}E\{(Y_i - \mu)(Y_2 - \mu)\} + ... + \frac{1}{n}E\{(Y_i - \mu)(Y_n - \mu)\}$$

This term is covariances between data points and we can rewrite it as

$$\frac{1}{n}Cov(Y_i, Y_1) + \frac{1}{n}Cov(Y_i, Y_2) + ... + \frac{1}{n}Cov(Y_i, Y_n)$$

If the data are independently sampled, the covariance between data points are $0$ except when the data points are the same $i=j$, in which case it is $Cov(Y_i, Y_j) = Cov(Y_i, Y_i) = \sigma^2$. Hence we get

$$E\{(Y_i-\mu)(\bar{Y} - \mu)\} = \frac{\sigma^2}{n}$$

so the biased variance would be

$$E(Y_i - \bar{Y}) = E\{(Y_i - \mu)^2\} + E\{(\bar{Y} - \mu)^2\} - 2E\{(Y_i - \mu)(\bar{Y} - \mu)\}$$

$$E(Y_i - \bar{Y}) = \sigma^2 + \frac{\sigma^2}{n} - 2\frac{\sigma^2}{n}$$

$$E(Y_i - \bar{Y}) = \sigma^2 - \frac{\sigma^2}{n}$$

$$E(Y_i - \bar{Y}) = \frac{(n - 1)}{n}\sigma^2$$

In order to correct this bias, or in another word for $E(Y_i - \bar{Y})$ to result in $\sigma^2$, we have to multiply this by $\frac{n}{n - 1}$. Hence, when we plug in this correction to our original variance equation, in which we use $n$, we get

$$\hat{\sigma}^2 = correction \cdot \frac{1}{n}\sum_{i} ^{}(Y_i - \bar{Y})^2$$

$$\hat{\sigma}^2 = \frac{n}{n-1} \cdot \frac{1}{n}\sum_{i} ^{}(Y_i - \bar{Y})^2$$

$$\hat{\sigma}^2 = \frac{1}{n-1}\sum_{i} ^{}(Y_i - \bar{Y})^2$$

This is the corrected, or in another word the unbiased, estimator of variance.

I was in my second year in university when I stumbled upon this equation during my lecture. At that time I was confused why we use $n-1$ instead of $n$ and ask that question to my lecturer. Maybe because either I am not smart enough or my lecturer can't explain this to me, I did not find the answer until 3 years later. This, in my opinion, is why we need estimation theory in undergraduate topic and we also need an understanding that statistics is trying to model real world process. And because it is a model, it can be wrong.
