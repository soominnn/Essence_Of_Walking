package com.walking.standardofwalking.dto;

import com.walking.standardofwalking.entity.User;
import lombok.AllArgsConstructor;
import lombok.ToString;

@AllArgsConstructor
@ToString
public class UserForm {

    private Long cid;
    private String name;
    private String userid;
    private String password;
    private String email;
    private String gender;
    private String age;
    private String phone;

    public User toEntity() { return new User(cid,name,userid,password,email,gender,age,phone); }
}
