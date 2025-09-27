# POLICY EVALUATION

## AIM
To evaluate and compare different policies in the Frozen Lake environment and find the best policy for reaching the goal successfully.

## PROBLEM STATEMENT
In the Frozen Lake environment, an agent must navigate from the start to the goal while avoiding holes. Movements are uncertain due to slipperiness. A policy guides the agent’s actions, but not all policies are effective. The task is to:

Evaluate a given policy (V1) using policy evaluation. Create and test a new policy (V2) to improve performance. Compare both policies based on success rate and rewards. Find the best policy for safely reaching the goal. This helps in identifying the most efficient way to complete the task.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)  # action chosen by the policy at state s
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V
```

## OUTPUT:
### POLICY 1:
<img width="674" height="192" alt="image" src="https://github.com/user-attachments/assets/61daa654-f805-4db4-92da-ef96e4cba10e" />
<img width="621" height="135" alt="image" src="https://github.com/user-attachments/assets/edf6d7ad-cca4-4a48-af87-0a648ce5b223" />

### POLICY 2:
<img width="674" height="192" alt="image" src="https://github.com/user-attachments/assets/61daa654-f805-4db4-92da-ef96e4cba10e" />
<img width="621" height="135" alt="image" src="https://github.com/user-attachments/assets/edf6d7ad-cca4-4a48-af87-0a648ce5b223" />

### COMPARISON:
<img width="684" height="241" alt="image" src="https://github.com/user-attachments/assets/a2b19ac8-2b9f-45d3-adf5-ad7b482aa8f5" />

## RESULT:

Thus, The Python program to evaluate the given policy is successfully executed.
