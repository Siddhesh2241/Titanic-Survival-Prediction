Create database if not exists new_titanic;

use new_titanic;

create table inter_titanicdata (
   ID int primary key auto_increment,
   Pclass int,
   Age int,
   SibSp int, 
   Parch int, 
   Fare float,
   Sex varchar(20),
   Embarked varchar(2),
   Predicition int 
   );
   
select * from inter_titanicdata;