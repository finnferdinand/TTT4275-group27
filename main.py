import classifiers.knn as cl
import utilities.plotting
from matplotlib import pyplot as plt

(confusion_matrix, misclassified, correctly_classified) = cl.classifier_1nn()
print(confusion_matrix)

utilities.plotting.draw_images(misclassified)
utilities.plotting.draw_images(correctly_classified)
plt.show()
