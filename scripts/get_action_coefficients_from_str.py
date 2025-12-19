import numpy as np

from transforms.actions import compute_action_coefficients

c = "[10.17165147 36.38967108  4.17807335  0.         54.26061079  6.37942382]"
c = map(float, c.replace("[", "").replace("]", "").split())
a = np.array(list(c))

b = compute_action_coefficients(a)
print(f"{b[0]:.2f} {b[1]:.2f} {b[2]:.2f}")
