package com.walking.standardofwalking.api;

import com.walking.standardofwalking.dto.UserForm;
import com.walking.standardofwalking.entity.User;
import com.walking.standardofwalking.service.UserService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@Slf4j
@RestController
public class UserController {
    @Autowired
    private UserService userService;

    // POST
    //유저 생성 -> 회원가입
    @PostMapping("/signup")
    public ResponseEntity<User> create(@RequestBody UserForm dto){
        User created = userService.create(dto);
        return (created != null) ?
                ResponseEntity.status(HttpStatus.OK).body(created):
                ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
    }

    //가입되어 있는 전체 회원정보 조회
    @GetMapping("/users/info")
    public List<User> getUser(){

        return userService.findAll();
    }

    //특정 id 회원정보 조회
    @GetMapping("/users/info/{userid}")
    public Optional<User> findByUserId(@PathVariable String userid){

        return userService.findByUserId(userid);
    }

    //회원정보 삭제
    @PostMapping("/users/delete/{userid}")
    public void deleteUser(@PathVariable String userid){
        userService.delete(userid);
    }
}
