package com.walking.standardofwalking.service;

import com.walking.standardofwalking.dto.LoginForm;
import com.walking.standardofwalking.entity.User;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.Optional;

@Service
@RequiredArgsConstructor

public class LoginService {
    @Autowired
    private final UserService userService;
    @Autowired
    PasswordEncoder passwordEncoder;

    public User login(String loginId, String loginPassword){

        Optional<User> user = userService.findByUserId(loginId);
        if(user.isPresent()){
            if(passwordEncoder.matches(loginPassword, user.get().getPassword())) {
                return user.get();
            }
        }
        return null;
    }
}
