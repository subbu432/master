from tkinter import*
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from tkinter import messagebox
import pandas as pd

def openFile():
    tf = filedialog.askopenfilename(
        title="Open CSV file",
        filetypes=(("CSV Files", "*.csv"),)
    )

    e1.insert(END,tf)
    tf = open(tf)
    global data
    data = pd.read_csv(tf)
    print(data)
    tf.close()

def checkdata():
    window.withdraw()
    global window2
    window2=Toplevel(window)
    window2.geometry("550x550")
    window2.configure(background='#454457')

    # creating the frame
    tree_frame = Frame(window2)
    tree_frame.pack(pady=50)

    # scroll bar
    tree_scroll = Scrollbar(tree_frame)
    tree_scroll.pack(expand=TRUE,side=RIGHT, fill=Y)
    tree_scroll_x = Scrollbar(tree_frame)
    tree_scroll_x.pack(expand=TRUE,side=BOTTOM, fill='both')

    # inserting scroll bar into the tree view
    tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, xscrollcommand=tree_scroll_x.set)
    tree.pack()

    # configure the scroll bar
    tree_scroll.configure(command=tree.yview)
    tree_scroll_x.configure(command=tree.xview)

    # to get the column values
    global c
    c = tuple(data.columns)
    print(c)
    tree["column"] = c


    # formating the column
    tree.column('#0', width=120)
    tree.heading('#0', text="s.no")
    for x in c:
        tree.column(x, width=120)
        # for creating column nmae


    for z in c:
        tree.heading(z, text=z)
    global sno
    sno = 0
    w = ((data.index))


    for lm in w:
        tree.insert(parent='', index="end", text=sno, values=tuple(data.iloc[lm, :]))

        sno += 1

    button = Button(window2, text="confirm", width=17, bg="#32a852", font=("times", 12, "bold"),
                    command=confirm)
    button.place(x=150,y=350)
    button = Button(window2, text="Reject", width=17, bg="#db2016", font=("times", 12, "bold"),
                    command=reject)
    button.place(x=350, y=350)



def confirm():
    window2.withdraw()
    global window3
    window3 = Toplevel(window)
    window3.geometry("550x550")
    window3.configure(background='#454457')



    lbl = Label(window3, text="select the independent variables",bg='#d0cade')

    listbox= Listbox(window3,selectmode="multiple")
    for x in c:
        y=1
        listbox.insert(1, x)
        y+=1
        lbl.pack()
        listbox.pack()
    def select():
         global predictor
         predictor=[listbox.get(i) for i in listbox.curselection()]
         print(predictor)
    buttonq=Button(window3,text="continue",bg='#32a852',command=select)
    buttonq.pack(pady=10)

    lbl2 = Label(window3, text="select the dependent variables",bg='#d0cade')

    listbox2 = Listbox(window3, selectmode="multiple")
    for x in c:
        y = 1
        listbox2.insert(1, x)
        y += 1
        lbl2.pack()
        listbox2.pack()

    def select():
        global target
        target = [listbox2.get(i) for i in listbox2.curselection()]
        print(len(target))
    buttonq = Button(window3, text="continue",bg='#32a852', command=select)
    buttonq.pack(pady=50)

    button = Button(window3, text=' confirm ', width=17, bg="#3de0cd", font=("times", 12, "bold"),
                    command=window4).place(x=400, y=500)

def window4():
    window3.withdraw()
    global window4
    window4= Toplevel(window)
    window4.geometry("550x550")
    window4.configure(background='#454457')

    label = Label(window4, text="Select the Algorithm", width=35, bg='#575444', fg="white",
                  font=("times", 15, "bold"))
    label.place(x=90, y=170)
    label.pack()

    list1 = ttk.Combobox(window4, value=['Simple Linear Regression', 'Multiple Linear Regression','polynomial Regression',
                                         'Logistic Regression',"K-NN Algorithm","Naive Bayes Algorithm","SVM","Decision Tree","Random Forest"])
    list1.place(x=50, y=50, height=80, width=230)
    list1.pack()
    def getting_the_Algorithm():
        global  Algo_type
        Algo_type = list1.get()
        global x,y,regressor
        if Algo_type == 'Simple Linear Regression':
            # import the data set
            for a in predictor:

                ad=data[a]
                ar=np.array(ad)
                x=np.reshape(ar,(len(ar),1))
                print(x)
            for b in target:
                ad2=data[b]

                ar2=np.array(ad2)
                y=np.reshape(ar2,(len(ar2),1))
                print(y)

            # split the data into training set and testing set

            from sklearn.model_selection import train_test_split

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1)

            # for building a model

            from sklearn.linear_model import LinearRegression
            global regressor
            regressor = LinearRegression()

            regressor.fit(x_train, y_train)




            # Prediction of Test and Training set result

            y_pred = regressor.predict(x_test)
            y_pred
            global acc
            acc=regressor.score(x_train, y_train)


        elif Algo_type == 'Multiple Linear Regression':
            for a in predictor:

                ad=data[a]
                ar=np.array(ad)
                x=np.reshape(ar,(len(ar),1))
                print(x)
            for b in target:
                ad2=data[b]

                ar2=np.array(ad2)
                y=np.reshape(ar2,(len(ar2),1))
                print(y)
            # Splitting the dataset into training and test set.
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            # Fitting the MLR model to the training set:
            from sklearn.linear_model import LinearRegression
            global regressor2
            regressor2 = LinearRegression()
            regressor2.fit(x_train, y_train)

            # Predicting the Test set result
            y_pred = regressor2.predict(x_test)
            global acc2
            acc2=regressor2.score(x_train, y_train)
            print(acc2)

        elif Algo_type =='polynomial Regression':
            for a in predictor:
                ad = data[a]
                ar = np.array(ad)
                x = np.reshape(ar, (len(ar), 1))
                print(x)
            for b in target:
                ad2 = data[b]

                ar2 = np.array(ad2)
                y = np.reshape(ar2, (len(ar2), 1))
                print(y)

            from sklearn.linear_model import LinearRegression
            lin_regs = LinearRegression()
            lin_regs.fit(x, y)

            # Fitting the Polynomial regression to the dataset
            from sklearn.preprocessing import PolynomialFeatures
            poly_regs = PolynomialFeatures(degree=3)
            x_poly = poly_regs.fit_transform(x)
            global lin_reg_2
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(x_poly, y)

        elif Algo_type =='Logistic Regression':
            for a in predictor:
                ad = data[a]
                ar = np.array(ad)
                x = np.reshape(ar, (len(ar), 1))
                print(x)
            for b in target:
                y = data[b]



            # Splitting the dataset into training and test set.
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

            # feature Scaling
            from sklearn.preprocessing import StandardScaler
            st_x = StandardScaler()
            x_train = st_x.fit_transform(x_train)
            x_test = st_x.transform(x_test)

            # Fitting Logistic Regression to the training set
            from sklearn.linear_model import LogisticRegression
            global classifierl
            classifierl= LogisticRegression(random_state=0)
            classifierl.fit(x_train, y_train)

            y_pred = classifier.predict(x_test)

            # Creating the Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)

            len(y_test)

            sum(np.diag(cm))
            global acc3
            acc3= (sum(np.diag(cm)) / len(y_test))



        elif Algo_type == "K-NN Algorithm":
            for a in predictor:
                ad = data[a]
                ar = np.array(ad)
                x = np.reshape(ar, (len(ar), 1))
                print(x)
            for b in target:
                y= data[b]


            # Splitting the dataset into training and test set.
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

            # feature Scaling
            from sklearn.preprocessing import StandardScaler
            st_x = StandardScaler()
            x_train = st_x.fit_transform(x_train)
            x_test = st_x.transform(x_test)

            # Fitting K-NN classifier to the training set
            from sklearn.neighbors import KNeighborsClassifier
            global classifierk
            classifierk = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
            classifierk.fit(x_train, y_train)

            # Predicting the test set result
            y_pred = classifierk.predict(x_test)

            # Creating the Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            len(y_test)

            sum(np.diag(cm))
            global acc4
            acc4= (sum(np.diag(cm)) / len(y_test))
            print(acc4)

        elif Algo_type =="Naive Bayes Algorithm":
            for a in predictor:
                ad = data[a]
                ar = np.array(ad)
                x = np.reshape(ar, (len(ar), 1))
                print(x)
            for b in target:
                y = data[b]

            # Splitting the dataset into the Training set and Test set
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

            # Feature Scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            # Fitting Naive Bayes to the Training set
            from sklearn.naive_bayes import GaussianNB
            global classifiern
            classifiern = GaussianNB()
            classifiern.fit(x_train, y_train)

            # Predicting the test set result
            y_pred = classifiern.predict(x_test)

            # Creating the Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            len(y_test)

            sum(np.diag(cm))
            global acc5
            acc5 = (sum(np.diag(cm)) / len(y_test))


        elif Algo_type =="SVM":
            for a in predictor:
                ad = data[a]
                ar = np.array(ad)
                x = np.reshape(ar, (len(ar), 1))
                print(x)
            for b in target:
                y = data[b]


            # Splitting the dataset into training and test set.
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
            # feature Scaling
            from sklearn.preprocessing import StandardScaler
            st_x = StandardScaler()
            x_train = st_x.fit_transform(x_train)
            x_test = st_x.transform(x_test)

            from sklearn.svm import SVC  # "Support vector classifier"
            global classifiers
            classifiers = SVC(kernel='linear', random_state=0)
            classifiers.fit(x_train, y_train)

            # Predicting the test set result
            y_pred = classifiers.predict(x_test)

            # Creating the Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            len(y_test)

            sum(np.diag(cm))
            global acc6
            acc6 = (sum(np.diag(cm)) / len(y_test))

        elif Algo_type =="Decision Tree":
            for a in predictor:
                ad = data[a]
                ar = np.array(ad)
                x = np.reshape(ar, (len(ar), 1))
                print(x)
            for b in target:
                y = data[b]


            # Splitting the dataset into training and test set.
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
            # Feature scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            # training the decisiontree classification model on the training set
            from sklearn.tree import DecisionTreeClassifier
            global classifierd
            classifierd = DecisionTreeClassifier(criterion='entropy', random_state=0)
            classifierd.fit(x_train, y_train)

            # Predicting the test set result
            y_pred = classifierd.predict(x_test)

            # Creating the Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            len(y_test)

            sum(np.diag(cm))
            global acc7
            acc7= (sum(np.diag(cm)) / len(y_test))

        elif Algo_type =="Random Forest":
            for a in predictor:
                ad = data[a]
                ar = np.array(ad)
                x = np.reshape(ar, (len(ar), 1))
                print(x)
            for b in target:
                y = data[b]


            # Splitting the dataset into training and test set.
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

            # Feature scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            # training the decisiontree classification model on the training set
            from sklearn.ensemble import RandomForestClassifier
            global classifierr
            classifierr = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5)
            classifierr.fit(x_train, y_train)


            # Predicting the test set result
            y_pred = classifierr.predict(x_test)

            # Creating the Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            len(y_test)

            sum(np.diag(cm))
            global acc8
            acc8 = (sum(np.diag(cm)) / len(y_test))


    model_creation = Button(window4, text="Train the model",bg="#3de0cd",command=getting_the_Algorithm).place(x=210, y=80, height=30, width=130)



    def lable_creation():
        window4.withdraw()
        window5 = Toplevel(window)
        window5.geometry("550x550")
        window5.configure(background='#454457')
        print(predictor)
        for c in range(len(predictor)):
            label = Label(window5, text=predictor[c], width=25, bg='#575444', fg="white",
                          font=("times", 15, "bold"))
            label.place(x=0, y=50 * c)



        entries=[]
        for d in range(len(predictor)):
            e= Entry(window5, width=20, font=("times", 10, "italic"))
            e.place(x=350, y=50 * d)
            entries.append(e)
        print(entries)
        def output():
           for entry in entries:
                global w
                w=entry.get()
                print(w)
                if Algo_type == 'Simple Linear Regression':
                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(w)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    y_pred = regressor.predict(input_data_reshaped)

                    print(y_pred)
                    messagebox.showinfo(target,y_pred)
                elif Algo_type ==  'Multiple Linear Regression':
                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(w)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    y_pred = regressor2.predict(input_data_reshaped)
                    print(y_pred)
                    messagebox.showinfo(target, y_pred)
                elif Algo_type ==  'Logistic Regression':
                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(w)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    y_pred = classifierl.predict(input_data_reshaped)
                    print(y_pred)
                    messagebox.showinfo(target,y_pred)
                elif Algo_type ==  'polynomial Regression':
                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(w)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    y_pred = lin_reg_2.predict(input_data_reshaped)
                    print(y_pred)
                    messagebox.showinfo(target, y_pred)
                elif Algo_type ==  "K-NN Algorithm":
                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(w)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    y_pred = classifierk.predict(input_data_reshaped)
                    print(y_pred)
                    messagebox.showinfo(target, y_pred)
                elif Algo_type ==  "Naive Bayes Algorithm":
                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(w)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    y_pred = classifiern.predict(input_data_reshaped)
                    print(y_pred)
                    messagebox.showinfo(target, y_pred)
                elif Algo_type ==  "SVM":
                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(w)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    y_pred = classifiers.predict(input_data_reshaped)
                    print(y_pred)
                    messagebox.showinfo(target, y_pred)
                elif Algo_type ==  "Decision Tree":
                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(w)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    y_pred = classifierd.predict(input_data_reshaped)
                    print(y_pred)
                    messagebox.showinfo(target, y_pred)
                elif Algo_type ==  "Random Forest":
                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(w)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                    y_pred = classifierr.predict(input_data_reshaped)
                    print(y_pred)
                    messagebox.showinfo(target, y_pred)
                    messagebox.configure(bg="#3de0cd")



        predict3= Button(window5, text="output", bg='#41a65c',command=output ).place(x=230, y=290)

        Accuracy = Button(window5, text="Accuracy of model", bg='#41a65c', command=Accuracy).place(x=200, y=180)


    predict= Button(window4, text="click to predict",bg='#41a65c',command=lable_creation).place(x=230, y=240)

def reject():
    window2.withdraw()
    window = Tk()
    window.geometry("550x550")
    window.configure(background='#454457')
    window.title('ML software for prediction and classification')

    label = Label(window, text="Select the Dataset to predict", width=35, bg='#575444', fg="white",
                  font=("times", 15, "bold"))
    label.place(x=90, y=170)

    e1 = Entry(window, width=50, font=("times", 10, "italic"))
    e1.place(x=90, y=230)

    button = Button(window, text="Browse", width=7, bg="#3de0cd", font=("times", 12, "bold"), command=openFile)
    button.place(x=450, y=220)

    button = Button(window, text="check the dataset", width=17, bg="#3de0cd", font=("times", 12, "bold"),
                    command=checkdata)
    button.place(x=350, y=270)

global window
window = Tk()
window.geometry("550x550")
window.configure(background='#454457')
window.title('ML software for prediction and classification')

label= Label(window, text = "Select the Dataset to predict",width=35,bg='#575444',fg="white",font=("times",15,"bold"))
label.place(x = 90,y =170)

e1=Entry(window,width=50,font=("times",10,"italic"))
e1.place(x=90,y=230)

button=Button(window,text="Browse",width=7,bg="#3de0cd",font=("times",12,"bold"),command=openFile)
button.place(x=450 ,y=220)

button=Button(window,text="check the dataset",width=17,bg="#3de0cd",font=("times",12,"bold"),command=checkdata)
button.place(x=350 ,y=270)

window.mainloop()


