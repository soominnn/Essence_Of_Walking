package com.walking.standardofwalking.service;

import com.walking.standardofwalking.dto.UserForm;
import com.walking.standardofwalking.entity.User;
import com.walking.standardofwalking.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;


@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    public User create(UserForm dto) {
        User user = dto.toEntity();
        //비밀번호 암호화
        String encodedPassword = passwordEncoder.encode(user.getPassword());
        user.setPassword(encodedPassword);

        if (user.getCid() != null) {
            return null;
        }
        return userRepository.save(user);
    }

    //아이디로 유저 찾기
    public Optional<User> findByUserId(String userid){
        return userRepository.findAll().stream()
                .filter(m -> m.getUserid().equals(userid))
                .findFirst();
    }
    public List<User> findAll(){
        return userRepository.findAll();
    }

    public void delete(String userid){
        Optional<User> user = findByUserId(userid);
        if(user.isPresent())
            userRepository.delete(user.get());
    }
}
