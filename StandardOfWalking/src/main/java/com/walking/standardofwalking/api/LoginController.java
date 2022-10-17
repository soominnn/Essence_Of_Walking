package com.walking.standardofwalking.api;

import com.walking.standardofwalking.dto.LoginForm;
import com.walking.standardofwalking.entity.User;
import com.walking.standardofwalking.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Optional;

@RestController
@RequestMapping("/login")
public class LoginController {

    @Autowired
    private UserService userService;
    @Autowired
    PasswordEncoder passwordEncoder;

    //로그인
    @PostMapping
    public String login(@RequestBody LoginForm form) {
        //입력받은 아이디와 패스워드
        String loginId = form.getUserid();
        String loginPass = form.getPassword();

       Optional<User> user =  userService.findByUserId(loginId);    //아이디로 유저 찾아서
       if(user.isPresent()){
           if(passwordEncoder.matches(loginPass, user.get().getPassword())){    //저장된 해시 비밀번호와 같으면 로그인 성공
               return "login success";
           }
       }
       return "login fail";
    }
}