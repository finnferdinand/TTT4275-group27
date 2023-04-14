import classifiers.knn as cl
import utilities.plotting
from matplotlib import pyplot as plt
import numpy as np

(confusion_matrix, misclassified, correctly_classified) = cl.classifier_1nn()
np.set_printoptions(suppress=True)
print("\nCONFUSION MATRIX:")
print(confusion_matrix)
print(f"SUCCESS RATE: {round(np.trace(confusion_matrix)/np.sum(confusion_matrix)*100,2)}%")

utilities.plotting.draw_images(misclassified)
utilities.plotting.draw_images(correctly_classified)
plt.show()
