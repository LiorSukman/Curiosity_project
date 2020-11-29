import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

chunksize = 200_000
b_size = 1
PIC_DIM = 28

# read the data, we use chunks as the data is large
tfr = pd.read_csv('data\generated_data.csv', chunksize = chunksize)

# the probability distribution: theta(x|s,a) = theta[s][a][x]
num_actions = 2 * (PIC_DIM ** 2) // (b_size ** 2) #num of actions is determined by the block size
theta = np.ones([10, num_actions, 10]) / 10.0 #initialize priors uniformly

# for each line in the data
D_KL = []
counter = 0
fit_ig = 0
non_fit_ig = 0
total_fit = 0
total_non_fit = 0
for df in tfr: #for each chunk
    for irow in df.iterrows():
        row = irow[1]

        counter += 1
        if counter % 100_000 == 0:
            print('finished processing %d rows' % counter)

        # integer of the numbers, to find the place in the discrete probability distribution
        n1 = int(row['s'])
        n2 = int(row['a']) - 1
        n3 = int(row['x'])

        fit = 1 if n1 == n3 else 0

        p_before = theta[n1, n2, :].flatten()

        # Bayesian updates
        sum_p = 0
        for t in range(10):
            p = 0.985 if t == n3 else 0.015 / 9 #this is based on the approximated accuracy of the model

            # theta[t|x] ~ theta[t] * theta[x]
            theta[n1, n2, t] *= p

            # this is for the normalization
            sum_p += theta[n1, n2, t]

        # normalization
        for t in range(10):
            theta[n1, n2, t] /= sum_p

        p_after = theta[n1, n2, :].flatten()

        D_KL_t = 0.0
        for t in range(10):
            if p_before[t] > 0.0 and p_after[t] > 0.0:
                D_KL_t += p_after[t] * np.log2(p_after[t] / p_before[t])
        D_KL.append(D_KL_t)

        total_fit += fit
        total_non_fit += 1 - fit
        fit_ig += D_KL_t if fit else 0
        non_fit_ig += 0 if fit else D_KL_t

print('total fitting examples: %d ; average information gain for fitting examples %f' % (total_fit, fit_ig / total_fit))
print('total non-fitting examples: %d ; average information gain for non-fitting examples %f' % (total_non_fit, non_fit_ig / total_non_fit))

#plot information gain graph
plt.plot(D_KL)
plt.title("Information Gain vs Example")
plt.xlabel("Examples")
plt.ylabel("Information Gain")
plt.show()

