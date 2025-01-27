import matplotlib.pyplot as plt
import numpy as np

img = np.array([[0, 128, 255],
                [255, 128, 0],
                [128, 0, 255]])

plt.imshow(img, cmap='gray')
#plt.show()




# Ahora en RGB

# Partes del esto, [red, green, blue]

img_rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    [[255, 255, 0], [0, 255, 255], [255, 0, 255]],
                    [[192, 192, 192], [128, 0, 128], [255, 165, 0]]])

plt.imshow(img_rgb)
plt.show()