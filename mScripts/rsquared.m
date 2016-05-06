function r = rsquared(yOrig, ypred, plot_) 

if nargin<3||isempty(plot_),
  plot_ =0;
end;

% we consider non NaN only
selected = ~isnan(yOrig) & ~isnan(ypred);
yOrig = yOrig(selected);
ypred= ypred(selected);

r = 1 - ( sum( (ypred-yOrig) .^ 2) / sum( (yOrig - mean(yOrig)) .^ 2) );

tit = sprintf('Response rate: %1.3f R-squared %1.3f\n', sum(selected)/numel(selected),r);

if plot_,
  plot(yOrig, ypred,'.');
  xlabel('Actual Sales');
  ylabel('Predicted Sales'); 
  hold on;
  a=axis;
  plot(a(1:2), a(1:2),'k--');
  title(tit);
else
  tit
end;