import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def on_mouse_press(event):
    global samples_x, samples_y
    global classify_state
    if event.button == 1 and classify_state != -1:
        x = round(event.xdata, 2)
        y = round(event.ydata, 2)
        cla = classify_state;
        print("click position:" ,event.button, x, y)
        new_point_x = np.array([[x, y]], dtype=np.float)
        new_point_y = np.array([[cla]], dtype=np.float)
        samples_x = np.append(samples_x, new_point_x, axis = 0)
        samples_y = np.append(samples_y, new_point_y, axis = 0)
        draw_point(new_point_x, new_point_y)

def on_key_press(event):
    global classify_state
    print(event.key)
    if event.key == '0':
        classify_state = 0
    elif event.key == '1':
        classify_state = 1
    elif event.key == '2':
        classify_state = 2        
    elif event.key == 'escape':
        classify_state = -1
    
    elif event.key == 'ctrl+z':
        plt.clf()
        draw_point(new_point_x, new_point_y)
        samples_data.append(np.hstack((new_point_x, new_point_y), axis=1))
    elif event.key == 'ctrl+a':
        save_samples('mysamples.csv')

def init_figure():
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('key_press_event', on_key_press)    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('samples points')
    
def draw_point(X, Y):
    index_0 = np.where(Y==0)
    plt.scatter(X[index_0,0],X[index_0,1],marker='x',color = 'b',label = '0',s = 15)
    index_1 =np.where(Y==1)
    plt.scatter(X[index_1,0],X[index_1,1],marker='o',color = 'r',label = '1',s = 15)
    index_2 =np.where(Y==2)
    plt.scatter(X[index_2,0],X[index_2,1],marker='v',color = 'g',label = '2',s = 15)
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
    else:
        samples_data = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1], [1, 3, 0], [2, 3, 0]], dtype=np.float)
            
    return samples_data;

classify_state = -1

samples_data = load_samples('mysamples.csv')
print(samples_data)

init_figure()
samples_x = samples_data[:,0:2]
samples_y = samples_data[:,2:3]
draw_point(samples_x, samples_y)
plt.legend(loc='lower right')
plt.show()
