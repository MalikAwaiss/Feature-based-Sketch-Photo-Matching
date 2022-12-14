#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.25.1
#  in conjunction with Tcl version 8.6
#    Oct 23, 2019 10:59:38 PM PKT  platform: Windows NT

import sys
import lgin
import register
from PIL import ImageTk, Image
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import first_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    root.state("zoomed")
    top = Toplevel1 (root)
    first_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt,im
    rt = root
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    first_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    def login(self):
#        root.withdraw()
        lgin.vp_start_gui()
    def register(self):
        root.withdraw()
        register.vp_start_gui()
        
        
        
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font10 = "-family {Segoe UI} -size 12 -weight bold -slant "  \
            "roman -underline 0 -overstrike 0"
        font9 = "-family {Segoe UI} -size 40 -weight bold -slant roman"  \
            " -underline 0 -overstrike 0"

        top.geometry("1451x1087+650+150")
        top.title("Sketch Matcher")
        top.configure(background="#e3faf9")
        
#        self.img.pack()

#        self.Label1 = tk.Label(top)
#        self.Label1.place(relx=0.240, rely=0.092, height=268, width=952)
#        self.Label1.configure(background="#e3faf9")
#        self.Label1.configure(disabledforeground="#a3a3a3")
#        self.Label1.configure(font=font9)
#        self.Label1.configure(foreground="#000000")
#        self.Label1.configure(text='''Sketch Matcher''')
        
        
        filenm ="logo.jpg"
        imgg=Image.open(filenm).resize((470,300))
        imgg = ImageTk.PhotoImage(imgg)
        self.img = tk.Label(top,image=imgg)
        self.img.image=imgg
        self.img.configure(background="#e3faf9")
#        img.image = imgg
        self.img.place(relx=0.390, rely=0.092, height=300, width=400)
#height=200, width=300///imgg=Image.open(filenm).resize((300,200))
        self.regbtn = tk.Button(top)
        self.regbtn.place(relx=0.368, rely=0.57, height=84, width=472)
        self.regbtn.configure(activebackground="#ececec")
        self.regbtn.configure(activeforeground="#000000")
        self.regbtn.configure(background="#ffffff")
        self.regbtn.configure(disabledforeground="#a3a3a3")
        self.regbtn.configure(font=font10)
        self.regbtn.configure(foreground="#000000")
        self.regbtn.configure(highlightbackground="#d9d9d9")
        self.regbtn.configure(highlightcolor="black")
        self.regbtn.configure(pady="0")
        self.regbtn.configure(font=("Courier", 20))
        self.regbtn.configure(text='''Register''')
        self.regbtn.configure(command=reg)

        self.lginbtn = tk.Button(top)
        self.lginbtn.place(relx=0.368, rely=0.432, height=84, width=472)
        self.lginbtn.configure(activebackground="#ececec")
        self.lginbtn.configure(activeforeground="#000000")
        self.lginbtn.configure(background="#ffffff")
        self.lginbtn.configure(disabledforeground="#a3a3a3")
        self.lginbtn.configure(font=font10)
        self.lginbtn.configure(foreground="#000000")
        self.lginbtn.configure(highlightbackground="#d9d9d9")
        self.lginbtn.configure(highlightcolor="black")
        self.lginbtn.configure(pady="0")
        self.lginbtn.configure(font=("Courier", 20))
        self.lginbtn.configure(text='''Login''')
        self.lginbtn.configure(command=loginn)
def loginn():
    global root
    root.destroy()
    lgin.vp_start_gui()
def reg():
    global root
    root.destroy()
    register.vp_start_gui()
if __name__ == '__main__':
    vp_start_gui()





