"""
Pseudocode details of the LSTM architecture

All three transformations have their own weight matrices, which you’ll index with the letters i, f, and k. Here’s what you have so far (it may seem a bit arbitrary, but bear with me).
"""
output_t = activation(dot(state_t, Uo) + dot(input_t, Wo) + dot(C_t, Vo) + bo)

i_t = activation(dot(state_t, Ui) + dot(input_t, Wi) + bi)
f_t = activation(dot(state_t, Uf) + dot(input_t, Wf) + bf)
k_t = activation(dot(state_t, Uk) + dot(input_t, Wk) + bk)
# You obtain the new carry state (the next c_t) by combining i_t, f_t, and k_t.
c_t+1 = i_t * k_t + c_t * f_t
