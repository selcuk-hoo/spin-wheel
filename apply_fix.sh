# Lines 266-286 arasında yeni kod
head -266 integrator.cpp.bak > integrator.cpp

cat >> integrator.cpp << 'CPPEOF'
    for (long long step = 0; step < total_steps; ++step) {
        gl4_step(t, y_init, field_params, h, dim);
        
        // dtheta: teğetsel momentum bileşeninden hesapla
        double X = y_init[0], Y = y_init[1];
        double Px = y_init[3], Py = y_init[4];
        double theta = std::atan2(Y, X);
        double R = std::sqrt(X*X + Y*Y);
        double p_tang = -Px * std::sin(theta) + Py * std::cos(theta);
        double p_total = std::sqrt(Px*Px + Py*Py + y_init[5]*y_init[5]);
        
        double dtheta = 0.0;
        if (R > 1e-3 && p_total > 1e-10) {
            dtheta = (p_tang / R) * (beta_ideal * C_LIGHT / p_total) * h;
        }
        double cur_theta_uw = prev_theta_uw + dtheta;
        t += h;

        // Calculate current FODO index
CPPEOF

tail -n +289 integrator.cpp.bak >> integrator.cpp
