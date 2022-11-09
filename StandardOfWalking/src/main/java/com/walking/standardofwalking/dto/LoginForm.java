package com.walking.standardofwalking.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.ToString;

import javax.validation.constraints.NotEmpty;

@AllArgsConstructor
@ToString
@Data
public class LoginForm {

    @NotEmpty
    private String userid;
    @NotEmpty
    private String password;
}
