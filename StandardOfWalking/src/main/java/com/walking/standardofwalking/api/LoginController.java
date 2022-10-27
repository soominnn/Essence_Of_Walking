package com.walking.standardofwalking.api;

import com.walking.standardofwalking.dto.LoginForm;
import com.walking.standardofwalking.entity.User;
import com.walking.standardofwalking.service.LoginService;
import com.walking.standardofwalking.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@RestController
public class LoginController {

    @Autowired
    private UserService userService;
    @Autowired
    private LoginService loginService;

    //서버에서 쿠키 생성
    @PostMapping("/login")
    public ResponseEntity login(@RequestBody LoginForm form, HttpServletResponse response){

        User loginUser = loginService.login(form.getUserid(), form.getPassword());  //아이디 비번 일치하는 회원 찾기

        if(loginUser == null){  //없을경우
            return new ResponseEntity<User>(loginUser, HttpStatus.NO_CONTENT);
        }

        //쿠키 생성
        Cookie idCookie = new Cookie("userid", String.valueOf(loginUser.getUserid()));
        response.addCookie(idCookie);

        return new ResponseEntity<User>(loginUser, HttpStatus.OK);
    }

    //서버에서 쿠키 조회하기
    @GetMapping("/")
    public ResponseEntity homeLogin(@CookieValue(name="userid", required = false) String userid){
        User loginUser = userService.findByUserId(userid).get();
        if(loginUser == null){
            return new ResponseEntity<User>(loginUser, HttpStatus.NO_CONTENT);
        }

        return new ResponseEntity<User>(loginUser, HttpStatus.OK);
    }

    @PostMapping("/logout")
    public ResponseEntity logout(HttpServletResponse response){
        Cookie cookie = new Cookie("userid", null);
        cookie.setMaxAge(0);
        response.addCookie(cookie);
        return new ResponseEntity(HttpStatus.OK);
    }

}