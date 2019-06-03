import numpy as np
import matplotlib.pyplot as plt
import os
import csv

SAMPLE_DATA_FILE = '../data/mysamples2_delta3.csv'
#OUTPUT_CLASS_NUM = 3
OUTPUT_CLASS_STYLES = np.array([
['0', 'o', 'r'], 
['1', 'x', 'g'], 
['2', 'v', 'b'],
['3', 's', 'k'],
['4', '*', 'y'],
['5', '^', '#808000'],
])
OUTPUT_CLASS_NUM = OUTPUT_CLASS_STYLES.shape[0]

def on_mouse_press(event):
    global samples_x, samples_y
    global classify_state
    if event.button == 1 and classify_state != -1: # add point
        print(event)
        x1 = round(event.xdata, 2)
        x2 = round(event.ydata, 2)
        cla = classify_state;
        print("click position:" ,event.button, x1, x2)
        new_point_x = np.array([[x1, x2]], dtype=np.float)
        new_point_y = np.array([[cla]], dtype=np.float)
        samples_x = np.append(samples_x, new_point_x, axis = 0)
        samples_y = np.append(samples_y, new_point_y, axis = 0)
        draw_point(new_point_x, new_point_y)
    elif event.button == 3 and samples_x.size > 0: # delete point
        x1 = round(event.xdata, 2)
        x2 = round(event.ydata, 2)
        target = np.array([[x1, x2]], dtype=np.float)
        d = list(map(lambda x: np.sum(np.square(x - target)), samples_x))
        if np.min(d) < 0.1:
            idx = d.index(np.min(d))
            samples_x = np.delete(samples_x, idx, 0)
            samples_y = np.delete(samples_y, idx, 0)
            redraw_all()

def on_key_press(event):
    global classify_state
    print(event.key)
    if len(event.key) == 1 and int(event.key) >= int('0') and int(event.key) <= int('9'):
        classify_state = int(event.key) - int('0')      
    elif event.key == 'escape':
        classify_state = -1    
    elif event.key == 'ctrl+z':
        redraw_all()
    elif event.key == 'ctrl+a':
        save_samples(SAMPLE_DATA_FILE)
        print('save file complete')

def init_figure():
    fig = plt.figure(figsize=(8, 6))
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('key_press_event', on_key_press)    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('sample points')
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))

def redraw_all():
    global samples_x, samples_y
    plt.clf()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('sample points')
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    draw_point(samples_x, samples_y)
    plt.legend(loc='lower right')


def draw_point(X, Y):
    for i in range(OUTPUT_CLASS_NUM):
        index = np.where(Y==i)[0]
        label = OUTPUT_CLASS_STYLES[i, 0]
        marker = OUTPUT_CLASS_STYLES[i, 1]
        color = OUTPUT_CLASS_STYLES[i, 2]
        plt.scatter(X[index, 0], X[index, 1], marker = marker, color = color, label = label, s = 30)

    plt.draw()

def save_samples(csv_file):
    global samples_x, samples_y
    samples_data = np.zeros(shape=(0, 3), dtype=np.float)
    samples_data = np.append(samples_data, np.hstack((samples_x, samples_y)), axis=0)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x1', 'x2', 'y'])
        for row in samples_data:
            writer.writerow(row)

def load_samples(csv_file):
    samples_data = np.zeros(shape=(0, 3), dtype=np.float)
    if (os.path.exists(csv_file)):
        with open(csv_file) as f:
            reader = csv.reader(f) #
            next(reader) #skip the header line
            for row in reader:
                sample = np.array(row)
                samples_data = np.append(samples_data, [sample.astype(np.float)], axis=0)
    #else:
        # samples_data = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1], [1, 3, 0], [2, 3, 0]], dtype=np.float)

    return samples_data;

classify_state = -1
samples_data = load_samples(SAMPLE_DATA_FILE)
init_figure()
samples_x = samples_data[:,0:2]
samples_y = samples_data[:,2:3]
print(np.hstack((samples_x, samples_y)))
redraw_all()
plt.show()
