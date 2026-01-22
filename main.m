%% MATLAB代码：绘制带元素标注的主角度示意图（可定制版）

% --- 1. 初始化 ---
clc;            % 清除命令行窗口
clear;          % 清除工作区变量
close all;      % 关闭所有图形窗口

% --- 2. 用户可修改参数 ---
fontSize = 25;      % <<-- 在这里统一修改所有标注的字体大小
outputFileName = 'Principal_Angles_Diagram.png'; % <<-- 在这里设置保存的文件名

% --- 3. 定义两个平面 ---
% 平面由其法向量定义
n_x = [0.5, -1, 1];   % 平面X的法向量
n_y = [0.5, 1, 1];    % 平面Y的法向量

% 将法向量单位化
n_x = n_x / norm(n_x);
n_y = n_y / norm(n_y);

% --- 4. 计算主向量和主角度 ---
% 第一个主向量 (x1, y1) 是交线方向
x1 = cross(n_x, n_y);
x1 = x1 / norm(x1); 
y1 = x1; % x1 和 y1 方向相同

% 第二个主向量 (x2, y2)
x2 = cross(x1, n_x);
x2 = x2 / norm(x2); 
y2 = cross(x1, n_y);
y2 = y2 / norm(y2);

% 第二个主角度 (theta2)
theta2 = acos(dot(x2, y2));

% --- 5. 可视化绘图 ---
fig = figure('Name', '主角度示意图', 'Color', 'w');
hold on;
axis equal; % 确保坐标轴比例相同
axis off;   % <<-- 移除所有坐标轴和边框
%view(    152.0262,  2.7833); % 设置视角

view( 115.3426, 31.4111); % 设置一个适合观察正交平面的视角



% 定义颜色
color_plane_x = '#0072BD'; % 蓝色
color_plane_y = '#D95319'; % 橙色
color_intersection = 'black'; % 交线颜色

% 创建网格来绘制平面
[x_grid, y_grid] = meshgrid(-2:0.5:2);
z_grid_x = -(n_x(1) * x_grid + n_x(2) * y_grid) / n_x(3);
z_grid_y = -(n_y(1) * x_grid + n_y(2) * y_grid) / n_y(3);

% 绘制两个半透明的平面
surf(x_grid, y_grid, z_grid_x, 'FaceColor', color_plane_x, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
surf(x_grid, y_grid, z_grid_y, 'FaceColor', color_plane_y, 'FaceAlpha', 0.4, 'EdgeColor', 'none');

% 绘制主向量
origin = [0, 0, 0];
vec_length = 2; % 向量箭头的长度
vec_width = 2.5; % 向量箭头的线宽

% 绘制交线向量 (x1, y1) - 使用您选择的黑色
quiver3(origin(1), origin(2), origin(3), x1(1), x1(2), x1(3), vec_length, 'Color', color_intersection, 'LineWidth', vec_width);

% 绘制平面X的第二个主向量
quiver3(origin(1), origin(2), origin(3), x2(1), x2(2), x2(3), vec_length, 'Color', color_plane_x, 'LineWidth', vec_width);

% 绘制平面Y的第二个主向量
quiver3(origin(1), origin(2), origin(3), y2(1), y2(2), y2(3), vec_length, 'Color', color_plane_y, 'LineWidth', vec_width);

% --- 6. 添加元素标注 ---
% 标注主向量 (使用fontSize变量)
text(x1(1)*vec_length*1.1, x1(2)*vec_length*1.1, x1(3)*vec_length*1.1, ...
    '{\bfu_1, v_1, \theta_1 = 0}', 'FontSize', fontSize, 'Color', 'k', 'FontWeight', 'bold');
text(x2(1)*vec_length*1.1, x2(2)*vec_length*1.1, x2(3)*vec_length*1.1, ...
    '{\bfu_2}', 'FontSize', fontSize, 'Color', color_plane_x, 'FontWeight', 'bold');
text(y2(1)*vec_length*1.1, y2(2)*vec_length*1.1, y2(3)*vec_length*1.1, ...
    '{\bfv_2}', 'FontSize', fontSize, 'Color', color_plane_y, 'FontWeight', 'bold');

% 标注角度 (使用fontSize变量)
% text(x1(1)*vec_length*1.1, x1(2)*vec_length*1.1, x1(3)*vec_length*1.1, '\theta_1 = 0', 'FontSize', fontSize, 'Color', 'k');

% 标注 theta2 (绘制一个弧线来表示)
arc_radius = 0.5;
angles = linspace(0, theta2, 50);
v_perp = y2 - dot(y2, x2) * x2; % 找到与x2正交且在x2,y2平面内的向量
v_perp = v_perp / norm(v_perp);
arc_points = arc_radius * (cos(angles)' * x2 + sin(angles)' * v_perp);
plot3(arc_points(:,1), arc_points(:,2), arc_points(:,3), 'k', 'LineWidth', 1.5);

% 在弧线旁边添加 theta2 文本
text_pos = arc_radius * 1.3 * (cos(theta2/2) * x2 + sin(theta2/2) * v_perp);
text(text_pos(1), text_pos(2), text_pos(3), '\theta_2', 'FontSize', fontSize, 'FontWeight', 'bold');

% --- 7. 完善视觉并保存文件 ---
%camlight head;    % 打光
%lighting phong;
hold off;

% 保存图形到文件
exportgraphics(gca, outputFileName, 'Resolution', 300);

fprintf('图形已保存为: %s\n', outputFileName);



%% MATLAB代码：绘制主角度较大（正交）的示意图 - v2箭头向上

% --- 1. 初始化 ---
%clc;            % 清除命令行窗口
%clear;          % 清除工作区变量
%close all;      % 关闭所有图形窗口

% --- 2. 用户可修改参数 ---
fontSize = 25;      % <<-- 在这里统一修改所有标注的字体大小
outputFileName = 'Principal_Angles_Large_Angle.png'; % <<-- 新的文件名

% --- 3. 定义两个互相垂直的平面 ---
% 定义 XY 平面 (法向量沿z轴) 和 XZ 平面 (法向量沿y轴)
n_x = [0, 0, 1];    % 平面X (蓝色, XY平面) 的法向量
n_y = [0, 1, 0];    % 平面Y (橙色, XZ平面) 的法向量

% 法向量已经是单位向量，无需单位化

% --- 4. 计算主向量和主角度 ---
% 第一个主向量 (v1, u1) 是交线方向 (x轴)
x1 = cross(n_x, n_y);
x1 = x1 / norm(x1); 
y1 = x1;

% 第二个主向量 (u2, v2)
x2 = cross(x1, n_x); % u2, 蓝色向量, 位于XY平面
x2 = x2 / norm(x2); 
% --- 主要修改点 ---
% 将原来的 cross(x1, n_y) 取反，使其指向z轴正方向（向上）
y2 = -cross(x1, n_y); % v2, 橙色向量, 位于XZ平面
y2 = y2 / norm(y2);

% 第二个主角度 (theta2) 仍然是 pi/2 (90度)
theta2 = acos(dot(x2, y2));

% --- 5. 可视化绘图 ---
fig = figure('Name', '主角度示意图 - v2向上', 'Color', 'w');
hold on;
axis equal; % 确保坐标轴比例相同
axis off;   % 移除所有坐标轴和边框
view( 115.3426, 31.4111); % 设置一个适合观察正交平面的视角

% 定义颜色
color_plane_x = '#0072BD'; % 蓝色 (u)
color_plane_y = '#D95319'; % 橙色 (v)
color_intersection = 'black'; 

% 创建网格来绘制平面
grid_range = -2:0.5:2;

% 绘制平面U (XY平面, z=0)
[x_grid_1, y_grid_1] = meshgrid(grid_range);
z_grid_1 = zeros(size(x_grid_1));
surf(x_grid_1, y_grid_1, z_grid_1, 'FaceColor', color_plane_x, 'FaceAlpha', 0.4, 'EdgeColor', 'none');

% 绘制平面V (XZ平面, y=0)
[x_grid_2, z_grid_2] = meshgrid(grid_range);
y_grid_2 = zeros(size(x_grid_2));
surf(x_grid_2, y_grid_2, z_grid_2, 'FaceColor', color_plane_y, 'FaceAlpha', 0.4, 'EdgeColor', 'none');

% 绘制主向量
origin = [0, 0, 0];
vec_length = 2; % 向量箭头的长度
vec_width = 2.5; % 向量箭头的线宽

% 绘制 v1, u1 (黑色交线)
quiver3(origin(1), origin(2), origin(3), x1(1), x1(2), x1(3), vec_length, 'Color', color_intersection, 'LineWidth', vec_width);
% 绘制 u2 (蓝色向量)
quiver3(origin(1), origin(2), origin(3), x2(1), x2(2), x2(3), vec_length, 'Color', color_plane_x, 'LineWidth', vec_width);
% 绘制 v2 (橙色向量, 现在指向上方)
quiver3(origin(1), origin(2), origin(3), y2(1), y2(2), y2(3), vec_length, 'Color', color_plane_y, 'LineWidth', vec_width);

% --- 6. 添加元素标注 ---
% 为了避免标签重叠，微调位置
text(x1(1)*vec_length*1.1, x1(2)*vec_length*1.05, x1(3)*vec_length*1.1, ...
    '{\bfv_1, u_1, \theta_1 = 0}', 'FontSize', fontSize, 'Color', 'k', 'FontWeight', 'bold');
text(x2(1)*vec_length*1.1, x2(2)*vec_length*1.1, x2(3)*vec_length*1.1, ...
    '{\bfu_2}', 'FontSize', fontSize, 'Color', color_plane_x, 'FontWeight', 'bold');
text(y2(1)*vec_length*0.8, y2(2)*vec_length*0.8, y2(3)*vec_length*1.1, ... % 调整v2标签位置
    '{\bfv_2}', 'FontSize', fontSize, 'Color', color_plane_y, 'FontWeight', 'bold');

% 标注 theta2 (绘制弧线)
arc_radius = 0.5;
angles = linspace(0, theta2, 50);
arc_points = arc_radius * (cos(angles)' * x2 + sin(angles)' * y2);
plot3(arc_points(:,1), arc_points(:,2), arc_points(:,3), 'k', 'LineWidth', 1.5);

text_pos = arc_radius * 1.3 * (cos(theta2/2) * x2 + sin(theta2/2) * y2);
text(text_pos(1), text_pos(2), text_pos(3), '\theta_2 = \pi/2', 'FontSize', fontSize, 'FontWeight', 'bold');

% --- 7. 完善视觉并保存文件 ---
hold off;
exportgraphics(gca, outputFileName, 'Resolution', 300);
fprintf('v2箭头向上的示意图已保存为: %s\n', outputFileName);