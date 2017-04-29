import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('vanilla.csv')

fig, ax1 = plt.subplots()
line1, = ax1.plot(range(len(df)), df['acc'], 'b-')
line2, = ax1.plot(range(len(df)), df['val_acc'], 'r-')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')

ax2 = ax1.twinx()
line3, = ax2.plot(range(len(df)), df['loss'], 'b--')
line4, = ax2.plot(range(len(df)), df['val_loss'], 'r--')
ax2.set_ylabel('Loss')
# fig.tight_layout()
plt.title('Learning Curve')
# Create a legend for the first line.
first_legend = plt.legend(handles=[line1, line2], loc=2)

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)

# Create another legend for the second line.
plt.legend(handles=[line3, line4], loc=3)
plt.savefig('task2_vanilla.png')
# plt.show()
plt.close()


df = pd.read_csv('dropOut.csv')

fig, ax1 = plt.subplots()
line1, = ax1.plot(range(len(df)), df['acc'], 'b-')
line2, = ax1.plot(range(len(df)), df['val_acc'], 'r-')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')

ax2 = ax1.twinx()
line3, = ax2.plot(range(len(df)), df['loss'], 'b--')
line4, = ax2.plot(range(len(df)), df['val_loss'], 'r--')
ax2.set_ylabel('Loss')
# fig.tight_layout()
plt.title('Learning Curve')
# Create a legend for the first line.
first_legend = plt.legend(handles=[line1, line2], loc=2)

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)

# Create another legend for the second line.
plt.legend(handles=[line3, line4], loc=3)
plt.savefig('task2_dropOut.png')
# plt.show()
plt.close()

