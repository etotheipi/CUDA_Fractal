% Take nx3 matrix of RGB values, return 256x3 rgb values
function map = createcolormap(colors, numpoints)

if(nargin < 2)
   numpoints = 256;
end

nc = size(colors,1);

x = 0:(nc-1);
xi = [0:(nc-1)/(numpoints-1):(nc-1)];

map = zeros(numpoints,3);
map(:,1) = interp1(x,colors(:,1),xi);
map(:,2) = interp1(x,colors(:,2),xi);
map(:,3) = interp1(x,colors(:,3),xi);
