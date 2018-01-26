"""
import sqlite3 as It
con = It.connect('student.db')
with con:
        cur = con.cursor() #pointer
       # cur.execute("create table Catch(moter1 int,moter2 int)")
        cur.execute("insert into Catch values(1,'A')")
       # cur.execute("insert into Catch values(2,'B')")
        #cur.execute("insert into Catch values(3,'C')")
        #cur.execute("insert into coms values(4,'D')")
con.close()

"""

import sqlite3
con = sqlite3.connect("Action2.db")


with con:
        cur = con.cursor()  # pointer
        """cur.execute('''CREATE TABLE Robot
                     (Action TEXT PRIMARY KEY,
                      S11 int ,
                      S12 int )''')"""
       # cur.execute("insert into Robot values('Ant2',1,3)")
        cur.execute("DELETE FROM Robot WHERE Action = 'Ant' ")
        Action2 = '\n'.join(con.iterdump())

con.close()