# point rendering

color zone #1 near #2 distance 20
vop splitbyzone #1


##  Gaussian
vop gaussian #3.2 sd 20

##  hide small object
surface dust #4 size 3000

## crop
只保留网格索引 x 10–110,y 20–120,z 5–55 区域
volume #1 region 10,20,5,110,120,55

## rotate
turn y 2 180 center #1

## view
view matrix

view matrix camera 0.74232,0.67005,-1.5042e-07,10254,-0.67005,0.74232,1.3577e-07,11061,2.0263e-07,1.9489e-15,1,31236