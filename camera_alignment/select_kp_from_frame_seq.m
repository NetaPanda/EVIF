% This code selects keypoints from a sequence of inf and vis images
% it then store the selected point pairs into input_points_array and
% base_points_array, which we can later save manually for computing warp matrix
% storing keypoints can let us calculate the warp matrix not only
% with matlab, but also with other libraries such as opencv

% usage: set the path and run
% a keypoint selection window will popup
% in the window, select keypoints from the left image, then select
% corresponding keypoints from the right window, repeat this operation to
% select at least 3 points
% open the File->export points to workspace, check all the checkbox
% (usually only the cpstruct checkbox is not checked, just check it) and
% click ok to export them to the workspace
% then click OK button in the "'Click OK after Export Points and cpstruct
% to Workspace from the File menu." dialog, the keypoint selection window
% will then close, and pop up again for the next pair of images, so that we
% can start over again to select more key points.
% if anything goes wrong, like you cant identify any keypoints in the
% current two images, just click ok button to close the window, the program
% will give an error and stop, but the selected keypoints are kept in the
% workspace and not lost, we can just manipulate the code (like change the
% value of current_i) to skip the problematic image pairs
% when we have selected enough image pairs, just export the input_points_array
% and base_points_array out (select both of them and export to one mat), 
% then we can calculate warp matrix with them


% in case we made something wrong in between and need to continue again, we dont clear critical
% variables
clearvars -except input_points_array base_points_array current_i current_idx;
clc;close all;


%% Input 2 images and imshow them;
% this will warp inf img towards the evt
inf_files = dir('C:\Users\gmy\Desktop\data\*_inf.png');
evt_files = dir('C:\Users\gmy\Desktop\data\*_evt.png');
% read the evt_files and inf_files
if ~exist('current_i', 'var')
    current_i = 100; % if the first few images are unclear, set current_i > 1, else it can be 1. Remember to clear all variables to make this valid
end
if ~exist('current_idx', 'var')
    current_idx = 1; % must start from 1
end

for i = current_i:300:length(inf_files)
    orthophoto = imread(['C:\Users\gmy\Desktop\data\' evt_files(i).name]);
    unregistered = imread(['C:\Users\gmy\Desktop\data\' inf_files(i).name]);
    unregistered = unregistered(:,:,1);
    [M,N]=size(unregistered);
    x=unregistered; % in this code, unregistered will be the modality that unchange, and the other modality will be warped towards this unregistered modality
    [m,n]=size(x); 
    orthophoto=imresize(orthophoto,[m n]);
    %% Give points and obtain the translation matrix; 
    h = cpselect(unregistered(:,:,1),orthophoto);
    uiwait(msgbox('Click OK after Export Points and cpstruct to Workspace from the File menu.','Waiting...'))
    close(h); % after click OK, we will close the kp selection window, just close the warning that pops up
    [input_points, base_points]=cpstruct2pairs(cpstruct);
    mytform = cp2tform(base_points, input_points,'affine ');
    disp(mytform.tdata.T);
    transMatrix = mytform.tdata.T; 
    % delete cpstruct variable
    clear cpstruct;
    % Save the input_points and base_points into corresponding arrays
    input_points_array{current_idx} = input_points;
    base_points_array{current_idx} = base_points;
    clear fixedPoints movingPoints;
    current_i = i;
    current_idx = current_idx + 1;
end







    
    
    
    
    
    
    
    
    
    





