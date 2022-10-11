package com.walking.standardofwalking.api;

import com.walking.standardofwalking.dto.UserForm;
import com.walking.standardofwalking.entity.User;
import com.walking.standardofwalking.service.UserService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
public class UserApiController {
    @Autowired
    private UserService userService;

    // POST
    @PostMapping("/api/users")
    public ResponseEntity<User> create(@RequestBody UserForm dto){
        User created = userService.create(dto);
        return (created != null) ?
                ResponseEntity.status(HttpStatus.OK).body(created):
                ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
    }
}
