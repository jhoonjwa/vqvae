'''
Hi. You will now embark on a new task. First, for a given .glb path, you will map each triangle's midpoint of the  │
│   3d mesh to the uv space.You must remember the 3D coordinates of each vertices that form the traingle: it will be   │
│   used later.  Then, find the furthest points for each side: like min(x), min(y), min(x), max(y), max(x), min(y),    │
│   max(x), max(y), and form a rectangle. In that rectangle, divide tha rectangle via 512 * 512 gird, and for each     │
│   grid, count how many midpoints are in there. Also you must keep the vertices of each triangle in 3d coords for     │
│   each traingle mapped in that grid. Afterwards, via that number of triangles mapped in a grid, we will apply        │
│   "pushing algorithm": see  At /home/ubuntu/jonghoon/mesh2mesh/vqvae/pushing_algorithm.py, it shows a minimal        │
│   example handling for a 2d grid. One thing to mind is that, we need to push while keeping the vertice's             │
│   coordinates: that means we need to keep a channel of 9, since triangle's vertices in a 3d space need 9             │
│   dimensions. The order of pushing should be sorted on the direction of the pushing: if we push to +x, the one with  │
│   the largest u should be pushed out first: ending in the rightmost position after pushing. It applies same for      │
│   -x, and +- y. If you are unsure of any of the techincal details, ask me and I will answer. Ultrathink and do your  │
│   very best. 
'''