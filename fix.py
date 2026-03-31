with open('integrator.cpp', 'r') as f:
    lines = f.readlines()

# Line 267-286 arasını bul ve değiştir
# Kolaylık: just replace the dtheta calculation lines

result = []
i = 0
while i < len(lines):
    if i == 272 and 'pz = y_init[5]' in lines[i]:
        # Buldum, 273-286 satırlarını skip et ve yeni kod ekle
        result.append(lines[266])  # gl4_step line
        result.append(lines[267])  # blank
        result.append("""        // dtheta hesaplanması: teğetsel momentum / R
        double X = y_init[0], Y = y_init[1];
        double Px = y_init[3], Py = y_init[4];  // GLOBAL momentum
        double theta = std::atan2(Y, X);
        double R = std::sqrt(X*X + Y*Y);
        double p_tangential = -Px * std::sin(theta) + Py * std::cos(theta);
        double p_total = std::sqrt(Px*Px + Py*Py + y_init[5]*y_init[5]);
        
        double dtheta = 0.0;
        if (R > 1e-3 && p_total > 1e-10) {
            dtheta = (p_tangential / R) * (beta_ideal * C_LIGHT / p_total) * h;
        }
        double cur_theta_uw = prev_theta_uw + dtheta;
        t += h;

""")
        # Skip eski kodu
        i += 1
        while i < len(lines) and 't += h;' not in lines[i]:
            i += 1
        i += 1  # t += h; satırını geç
        continue
    
    result.append(lines[i])
    i += 1

with open('integrator.cpp', 'w') as f:
    f.writelines(result)

print("Fixed!")
