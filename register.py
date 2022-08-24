#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.25.1
#  in conjunction with Tcl version 8.6
#    Oct 23, 2019 10:37:10 PM PKT  platform: Windows NT

import sys
import t1
import lgin
import first
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    from tkinter import messagebox
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    
    from tkinter import messagebox
    py3 = True

import register_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    root.state("zoomed")
    top = register (root)
    register_support.init(root, top)
    root.mainloop()

w = None
def create_register(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    top = register (w)
    register_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_register():
    global w
    
    w.destroy()
    w = None

class register:
    def back(self):
        root.withdraw()
        first.vp_start_gui()
        
    def register(self):
        file="cred.txt"
        user=self.userent.get()
        pw=self.pwent.get()
        cni=self.cninc.get()
        if user=="" or pw=="" or cni=="":
            messagebox.showerror("Registration Failed","User must provide all Fields.")
        else:
            self.userent.delete(0,'end')
            self.pwent.delete(0,'end')
            self.cninc.delete(0,'end')
            file=open(file,"a")
            file.write(user + "\n")
            file.write(pw + "\n")
            file.write(cni + "\n")
            file.close()
            self.Label1.configure(text="Registration Successful!")
    #        root.withdraw()
    #        first.vp_start_gui()
            regsuc()
        
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font11 = "-family {Segoe UI} -size 10 -weight bold -slant "  \
            "roman -underline 0 -overstrike 0"
        font9 = "-family {Segoe UI} -size 40 -weight bold -slant roman"  \
            " -underline 0 -overstrike 0"

        top.geometry("910x838+650+150")
        top.title("Sketch Maker")
        top.configure(background="#e3faf9")

        self.Label1 = tk.Label(top)
        self.Label1.place(relx=0.33, rely=0.072, height=218, width=652)
        self.Label1.configure(background="#e3faf9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font=font9)
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Registration''')

        self.userent = tk.Entry(top)
        self.userent.place(relx=0.385, rely=0.453,height=34, relwidth=0.356)
        self.userent.configure(background="white")
        self.userent.configure(disabledforeground="#a3a3a3")
        self.userent.configure(font="TkFixedFont")
        self.userent.configure(foreground="#000000")
        self.userent.configure(insertbackground="black")

        self.pwent = tk.Entry(top)
        self.pwent.place(relx=0.385, rely=0.549,height=34, relwidth=0.356)
        self.pwent.configure(background="white")
        self.pwent.configure(disabledforeground="#a3a3a3")
        self.pwent.configure(font="TkFixedFont")
        self.pwent.configure(foreground="#000000")
        self.pwent.configure(insertbackground="black")

        self.cninc = tk.Entry(top)
        self.cninc.place(relx=0.385, rely=0.649,height=34, relwidth=0.356)
        self.cninc.configure(background="white")
        self.cninc.configure(disabledforeground="#a3a3a3")
        self.cninc.configure(font="TkFixedFont")
        self.cninc.configure(foreground="#000000")
        self.cninc.configure(insertbackground="black")


        self.Label2 = tk.Label(top)
        self.Label2.place(relx=0.187, rely=0.453, height=38, width=125)
        self.Label2.configure(background="#e3faf9")
        self.Label2.configure(disabledforeground="#3b5998")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(font=("Courier", 18))
        self.Label2.configure(text='''Username:''')

        self.Label3 = tk.Label(top)
        self.Label3.place(relx=0.187, rely=0.549, height=38, width=120)
        self.Label3.configure(background="#e3faf9")
        self.Label3.configure(disabledforeground="#a3a3a3")
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(font=("Courier", 18))
        self.Label3.configure(text='''Password:''')

        self.Label33 = tk.Label(top)
        self.Label33.place(relx=0.187, rely=0.645, height=38, width=120)
        self.Label33.configure(background="#e3faf9")
        self.Label33.configure(disabledforeground="#a3a3a3")
        self.Label33.configure(foreground="#000000")
        self.Label33.configure(font=("Courier", 18))
        self.Label33.configure(text='''CNIC:''')


        self.Label4 = tk.Label(top)
        self.Label4.place(relx=0.38, rely=0.334, height=43, width=461)
        self.Label4.configure(background="#e3faf9")
        self.Label4.configure(disabledforeground="#a3a3a3")
        self.Label4.configure(font=("TkFixedFont",12))
        self.Label4.configure(foreground="#000000")
        self.Label4.configure(text='''Enter Your Username and Password''')

        self.Button1 = tk.Button(top)
        self.Button1.place(relx=0.657, rely=0.744, height=54, width=162)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#FFFFFF")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(font=("Courier", 12))
        self.Button1.configure(text='''Register''')
        self.Button1.configure(command=self.register)

        self.bac = tk.Button(top)
        self.bac.place(relx=0.657, rely=0.84, height=54, width=162)
        self.bac.configure(activebackground="#ececec")
        self.bac.configure(activeforeground="#000000")
        self.bac.configure(background="#FFFFFF")
        self.bac.configure(disabledforeground="#a3a3a3")
        self.bac.configure(foreground="#000000")
        self.bac.configure(highlightbackground="#d9d9d9")
        self.bac.configure(highlightcolor="black")
        self.bac.configure(pady="0")
        self.bac.configure(font=("Courier", 12))
        self.bac.configure(text='''Back''')
        self.bac.configure(command=backk)

        self.Label5 = tk.Label(top)
        self.Label5.place(relx=0.1, rely=0.811, height=118, width=802)
        self.Label5.configure(background="#e3faf9")
        self.Label5.configure(disabledforeground="#a3a3a3")
        self.Label5.configure(foreground="#000000")
        self.Label5.configure(font=("TkFixedFont",12))
        self.Label5.configure(text='''Note: Username and password can contain alphabets and numbers only.\n Note: CNIC is required for Password Recovery.''')
def regsuc():
    global root
    root.destroy()
    first.vp_start_gui()
def backk():
    global root
    root.destroy()
    first.vp_start_gui()
if __name__ == '__main__':
    vp_start_gui()





