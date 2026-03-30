# ============================================# 
# 論文用 Figure 生成（Fig. 1, 2, 3, 4 個別出力）
# ※ 完全連成モデルの実行後に実行すること
# ============================================

# === Fig. 1: 空間プロファイル ===
fig1, ax1 = plt.subplots(figsize=(8, 6))

x_plot = np.linspace(0, 1, 200)
x_input = np.column_stack([x_plot, np.full(200, 1.0)])
y_out = model.predict(x_input)
C_pred = y_out[:, 0].flatten() * C_range + C_surface
x_um = x_plot * L_depth * 1e4

ax1.plot(x_um, C_pred, 'b-', linewidth=2, 
         label=f'Coupled PINN ($D_{{0,PINN}}$ = {D_base:.1e} cm²/s)')
ax1.scatter(zheng_x_wall, zheng_Cr, c='red', s=20, alpha=0.5, 
            label='Zheng et al. [7] (EDS)')
ax1.set_xlabel('Depth from wall surface (μm)', fontsize=12)
ax1.set_ylabel('Cr concentration (wt%)', fontsize=12)
ax1.legend(fontsize=10)
ax1.set_xlim(0, 80)
ax1.set_ylim(0, 22)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Fig1_spatial_profile.png', dpi=300, bbox_inches='tight')
plt.show()
print("Fig. 1 saved: Fig1_spatial_profile.png")

# === Fig. 2: 質量変化 ===
fig2, ax2 = plt.subplots(figsize=(8, 6))

t_check_hr = np.array([1000, 2000, 3000])
t_check_norm = t_check_hr * 3600 / T_max_s

x_query = np.ones((3, 1))
t_query = t_check_norm.reshape(3, 1)
query_points = np.hstack([x_query, t_query])
y_pred_mass = model.predict(query_points)
w_pred = y_pred_mass[:, 1]
dW_predicted = w_pred * conversion_factor

dW_measured_avg = np.array([
    np.mean(dW_data[:2]), np.mean(dW_data[2:4]), np.mean(dW_data[4:])
])
dW_measured_err = np.array([
    (dW_data[1]-dW_data[0])/2, (dW_data[3]-dW_data[2])/2, (dW_data[5]-dW_data[4])/2
])

ax2.errorbar(t_check_hr, dW_measured_avg, yerr=dW_measured_err,
             fmt='ro', markersize=8, capsize=5, capthick=1.5,
             label='Measured [12]', zorder=5)
ax2.plot(t_check_hr, dW_predicted, 'b-s', linewidth=2, markersize=8,
         label='Coupled PINN')
ax2.set_xlabel('Exposure time (h)', fontsize=12)
ax2.set_ylabel('Weight loss (mg/cm²)', fontsize=12)
ax2.legend(fontsize=10)
ax2.set_xlim(0, 3500)
ax2.set_ylim(0, 0.7)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Fig2_mass_loss.png', dpi=300, bbox_inches='tight')
plt.show()
print("Fig. 2 saved: Fig2_mass_loss.png")

# === Fig. 3: 時間発展 ===
fig3, ax3 = plt.subplots(figsize=(8, 6))

times_hr = [0, 500, 1000, 2000, 3000]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for t_hr, color in zip(times_hr, colors):
    t_n = t_hr * 3600 / T_max_s
    x_input = np.column_stack([x_plot, np.full(200, t_n)])
    y_out_t = model.predict(x_input)
    C_t = y_out_t[:, 0].flatten() * C_range + C_surface
    ax3.plot(x_um, C_t, color=color, linewidth=2, label=f't = {t_hr} h')

ax3.scatter(zheng_x_wall, zheng_Cr, c='red', s=15, alpha=0.3,
            label='EDS data (3000 h)', zorder=1)
ax3.set_xlabel('Depth from wall surface (μm)', fontsize=12)
ax3.set_ylabel('Cr concentration (wt%)', fontsize=12)
ax3.set_xlim(0, 80)
ax3.set_ylim(0, 22)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.annotate('', xy=(25, 3), xytext=(5, 3),
             arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax3.text(15, 2, 'Depletion front\nadvances', fontsize=9,
         ha='center', color='gray', style='italic')

plt.tight_layout()
plt.savefig('Fig3_time_evolution.png', dpi=300, bbox_inches='tight')
plt.show()
print("Fig. 3 saved: Fig3_time_evolution.png")

# === Fig. 4: 学習損失 ===
fig4, ax4 = plt.subplots(figsize=(8, 5))

loss_train = np.array(losshistory.loss_train)
steps = np.array(losshistory.steps)
total_loss = np.sum(loss_train, axis=1)

ax4.semilogy(steps, total_loss, 'b-', linewidth=1.5, label='Total loss')

if loss_train.shape[1] >= 2:
    ax4.semilogy(steps, loss_train[:, 0], '--', color='#ff7f0e',
                 linewidth=1, alpha=0.7, label='PDE residual (diffusion)')
    ax4.semilogy(steps, loss_train[:, 1], '--', color='#2ca02c',
                 linewidth=1, alpha=0.7, label='PDE residual (integral)')

ax4.set_xlabel('Training iterations', fontsize=12)
ax4.set_ylabel('Loss', fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('Fig4_training_loss.png', dpi=300, bbox_inches='tight')
plt.show()
print("Fig. 4 saved: Fig4_training_loss.png")

print("\n=== 全Figure生成完了 ===")
print("Fig1_spatial_profile.png  → 論文 Fig. 1")
print("Fig2_mass_loss.png        → 論文 Fig. 2")
print("Fig3_time_evolution.png   → 論文 Fig. 3")
print("Fig4_training_loss.png    → 論文 Fig. 4")
