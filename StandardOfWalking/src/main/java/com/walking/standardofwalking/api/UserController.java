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
//@CrossOrigin(origins = "http://localhost:8080")
@RestController
public class UserController {
    @Autowired
    private UserService userService;

    // POST
    //유저 생성 -> 회원가입
    @PostMapping("/users")
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
    @DeleteMapping("/users/delete/{userid}")
    public ResponseEntity<User> delete(@PathVariable String userid){
        Optional<User> deleted = userService.delete(userid);
        return (deleted != null) ?
                ResponseEntity.status(HttpStatus.NO_CONTENT).build() :
                ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
    }

    //회원정보 수정
    @PatchMapping("/users/info/{userid}")
    public ResponseEntity<User> update(@PathVariable String userid, @RequestBody UserForm dto) {
        User updated = userService.update(userid, dto);
        return (updated != null) ?
                ResponseEntity.status(HttpStatus.OK).body(updated) :
                ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
    }
}
