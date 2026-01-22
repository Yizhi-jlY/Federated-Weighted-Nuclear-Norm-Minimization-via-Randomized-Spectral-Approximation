%% MATLAB代码：为论文生成主角度对比图
%
% 功能：
% 1. 生成一个主角度较小 (well-aligned) 的示意图。
% 2. 生成一个主角度较大 (poorly-aligned) 的示意图。
%
% 特点：
% - 蓝色平面 (Subspace U) 在两张图中完全一致。
% - 观察视角 (Camera View) 在两张图中完全一致，便于对比。

% --- 1. 初始化 ---
clc;            % 清除命令行窗口
clear;          % 清除工作区变量
close all;      % 关闭所有图形窗口

% --- 2. 通用参数设置 ---
fontSize = 25;      % 所有标注的字体大小
vec_length = 2;     % 向量箭头的长度
vec_width = 2.5;    % 向量箭头的线宽
common_view = [120, 25]; % <<-- 统一的观察视角 [方位角, 仰角]

% 定义颜色
color_plane_u = '#0072BD'; % 蓝色 (Subspace U)
color_plane_v = '#D95319'; % 橙色 (Subspace V)
color_intersection = 'black'; 

% --- 3. 定义公共的蓝色平面 (Subspace U) ---
% 为了简单和一致，我们使用 XY 平面作为公共的蓝色平面
% 其法向量为 Z 轴方向
n_u = [0, 0, 1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 图 1: 生成主角度较小的示意图 (Well-Aligned Subspaces)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- 定义一个与蓝色平面夹角很小的橙色平面 ---
n_v_small = [0.2, 0, 1]; % 法向量与 n_u 非常接近
n_v_small = n_v_small / norm(n_v_small); % 单位化

% --- 计算主向量和主角度 ---
u1 = cross(n_u, n_v_small); u1 = u1/norm(u1);
v1 = u1;
u2 = cross(u1, n_u); u2 = u2/norm(u2);
v2_small = cross(u1, n_v_small); v2_small = v2_small/norm(v2_small);
theta2_small = acos(dot(u2, v2_small));

% --- 绘图 ---
figure('Name', '小角度示意图', 'Color', 'w');
hold on;

% 创建网格并绘制平面
[x_grid, y_grid] = meshgrid(-2:0.5:2);
z_grid_u = zeros(size(x_grid)); % 蓝色平面是 XY 平面 (z=0)
z_grid_v_small = -(n_v_small(1) * x_grid + n_v_small(2) * y_grid) / n_v_small(3);
surf(x_grid, y_grid, z_grid_u, 'FaceColor', color_plane_u, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
surf(x_grid, y_grid, z_grid_v_small, 'FaceColor', color_plane_v, 'FaceAlpha', 0.4, 'EdgeColor', 'none');

% 绘制主向量
origin = [0, 0, 0];
quiver3(origin(1), origin(2), origin(3), u1(1), u1(2), u1(3), vec_length, 'Color', color_intersection, 'LineWidth', vec_width);
quiver3(origin(1), origin(2), origin(3), u2(1), u2(2), u2(3), vec_length, 'Color', color_plane_u, 'LineWidth', vec_width);
quiver3(origin(1), origin(2), origin(3), v2_small(1), v2_small(2), v2_small(3), vec_length, 'Color', color_plane_v, 'LineWidth', vec_width);

% 添加标注
text(u1(1)*vec_length*1.1, u1(2)*vec_length*1.1, u1(3)*vec_length*1.1, ...
    '{\bfu_1, v_1, \theta_1 = 0}', 'FontSize', fontSize, 'Color', 'k', 'FontWeight', 'bold');
text(u2(1)*vec_length*1.1, u2(2)*vec_length*1.1, u2(3)*vec_length*1.1, ...
    '{\bfu_2}', 'FontSize', fontSize, 'Color', color_plane_u, 'FontWeight', 'bold');
text(v2_small(1)*vec_length*1.1, v2_small(2)*vec_length*1.1, v2_small(3)*vec_length*1.1, ...
    '{\bfv_2}', 'FontSize', fontSize, 'Color', color_plane_v, 'FontWeight', 'bold');

% 标注角度 θ₂
arc_radius = 0.6;
angles = linspace(0, theta2_small, 50);
v_perp = v2_small - dot(v2_small, u2) * u2; v_perp = v_perp / norm(v_perp);
arc_points = arc_radius * (cos(angles)' * u2 + sin(angles)' * v_perp);
plot3(arc_points(:,1), arc_points(:,2), arc_points(:,3), 'k', 'LineWidth', 1.5);
text_pos = arc_radius * 1.4 * (cos(theta2_small/2) * u2 + sin(theta2_small/2) * v_perp);
text(text_pos(1), text_pos(2), text_pos(3), '\theta_2 \approx 0', 'FontSize', fontSize, 'FontWeight', 'bold');

% --- 设置视角并保存 ---
axis equal; axis off;
view(common_view); % <<-- 使用统一视角
hold off;
outputFileName_small = 'Principal_Angles_Small_Angle.png';
exportgraphics(gca, outputFileName_small, 'Resolution', 300);
fprintf('小角度示意图已保存为: %s\n', outputFileName_small);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 图 2: 生成主角度较大的示意图 (Poorly-Aligned Subspaces)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- 定义一个与蓝色平面垂直的橙色平面 ---
n_v_large = [0, 1, 0]; % 法向量与 n_u 正交 (XZ 平面)

% --- 计算主向量和主角度 ---
u1 = cross(n_u, n_v_large); u1 = u1/norm(u1);
v1 = u1;
u2 = cross(u1, n_u); u2 = u2/norm(u2);
v2_large = -cross(u1, n_v_large); v2_large = v2_large/norm(v2_large); % 负号确保 v2 向上
theta2_large = acos(dot(u2, v2_large)); % 应为 pi/2

% --- 绘图 ---
figure('Name', '大角度示意图', 'Color', 'w');
hold on;

% 创建网格并绘制平面
[x_grid, y_grid] = meshgrid(-2:0.5:2);
z_grid_u = zeros(size(x_grid)); % 蓝色平面是 XY 平面 (z=0)
[x_grid_2, z_grid_2] = meshgrid(-2:0.5:2);
y_grid_2 = zeros(size(x_grid_2)); % 橙色平面是 XZ 平面 (y=0)
surf(x_grid, y_grid, z_grid_u, 'FaceColor', color_plane_u, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
surf(x_grid_2, y_grid_2, z_grid_2, 'FaceColor', color_plane_v, 'FaceAlpha', 0.4, 'EdgeColor', 'none');

% 绘制主向量
quiver3(origin(1), origin(2), origin(3), u1(1), u1(2), u1(3), vec_length, 'Color', color_intersection, 'LineWidth', vec_width);
quiver3(origin(1), origin(2), origin(3), u2(1), u2(2), u2(3), vec_length, 'Color', color_plane_u, 'LineWidth', vec_width);
quiver3(origin(1), origin(2), origin(3), v2_large(1), v2_large(2), v2_large(3), vec_length, 'Color', color_plane_v, 'LineWidth', vec_width);

% 添加标注
text(u1(1)*vec_length*1.1, u1(2)*vec_length*1.05, u1(3)*vec_length*1.1, ...
    '{\bfu_1, v_1, \theta_1 = 0}', 'FontSize', fontSize, 'Color', 'k', 'FontWeight', 'bold');
text(u2(1)*vec_length*1.1, u2(2)*vec_length*1.1, u2(3)*vec_length*1.1, ...
    '{\bfu_2}', 'FontSize', fontSize, 'Color', color_plane_u, 'FontWeight', 'bold');
text(v2_large(1)*vec_length*0.8, v2_large(2)*vec_length*0.8, v2_large(3)*vec_length*1.1, ...
    '{\bfv_2}', 'FontSize', fontSize, 'Color', color_plane_v, 'FontWeight', 'bold');

% 标注角度 θ₂
arc_radius = 0.6;
angles = linspace(0, theta2_large, 50);
arc_points = arc_radius * (cos(angles)' * u2 + sin(angles)' * v2_large);
plot3(arc_points(:,1), arc_points(:,2), arc_points(:,3), 'k', 'LineWidth', 1.5);
text_pos = arc_radius * 1.3 * (cos(theta2_large/2) * u2 + sin(theta2_large/2) * v2_large);
text(text_pos(1), text_pos(2), text_pos(3), '\theta_2 = \pi/2', 'FontSize', fontSize, 'FontWeight', 'bold');

% --- 设置视角并保存 ---
axis equal; axis off;
view(common_view); % <<-- 使用统一视角
hold off;
outputFileName_large = 'Principal_Angles_Large_Angle.png';
exportgraphics(gca, outputFileName_large, 'Resolution', 300);
fprintf('大角度示意图已保存为: %s\n', outputFileName_large);