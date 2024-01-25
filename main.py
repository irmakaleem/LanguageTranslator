import tkinter as tk
from tkinter import ttk
from googletrans import Translator, LANGUAGES

root = tk.Tk()
root.geometry('700x300')
root.resizable(0, 0)
root['bg'] = '#FAFAFA'  # Light gray background

root.title('Language Translator')  
tk.Label(root, text="Language Translator", font="Arial 20 bold", fg='#212121').pack()

# Input text field
tk.Label(root, text="Enter text:", font="Arial 14 bold", fg='#212121').place(x=20, y=60)
input_text = tk.Entry(root, width=50, font="Arial 12", fg='#212121', bg='#F2F2F2')
input_text.place(x=130, y=60)

# Output text field
tk.Label(root, text="Output:", font="Arial 14 bold", fg='#212121').place(x=20, y=150)
output_text = tk.Text(root, width=50, height=1, font="Arial 12", fg='#212121', bg='#F2F2F2')
output_text.place(x=130, y=150)

# Destination language selection
languages = list(LANGUAGES.values())
dest_lang = ttk.Combobox(root, values=languages, width=20, font="Arial 12", foreground='#212121')
dest_lang.pack(side=tk.TOP, pady=20)
dest_lang.place(x=300, y=110)
dest_lang.set('Choose language')


def Translate():
    translator = Translator()
    translated = translator.translate(text=input_text.get(), dest=dest_lang.get())
    output_text.delete('1.0', tk.END)
    output_text.insert('1.0', translated.text)
# Translate button
translate_btn = tk.Button(root, text='Translate', font="Arial 12 bold", bg='#007bff', fg='white', activebackground='#0069d9', activeforeground='white', command=Translate)
translate_btn.place(relx=0.5, y=230, anchor='center')

root.mainloop()
