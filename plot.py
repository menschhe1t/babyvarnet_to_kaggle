import h5py
import matplotlib.pyplot as plt

f = h5py.File('/kaggle/working/result/test_Unet/reconstructions_val/brain_acc4_179.h5', 'r')

print(f)
input = f['input']
recon = f['reconstruction']
target = f['target']

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(input[1, :, :])
plt.title('input')
plt.subplot(1, 3, 2)
plt.imshow(recon[1, :, :])
plt.title('reconstruction')
plt.subplot(1, 3, 3)
plt.imshow(target[1, :, :])
plt.title('target')
plt.savefig('result.png', dpi=300)

##
plt.imshow()
